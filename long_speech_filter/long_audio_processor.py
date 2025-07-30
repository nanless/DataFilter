"""
长音频处理主模块
整合说话人分离、音频分割和质量筛选功能
基于现有speech_filter框架
"""
import os
import json
import logging
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import LongAudioProcessingConfig
from speaker_diarization import LongAudioSpeakerDiarizer
from quality_filter import LongAudioQualityFilter

# 设置日志
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """单个长音频的处理结果"""
    audio_path: str
    audio_id: str 
    success: bool
    speaker_count: int = 0
    total_segments: int = 0
    passed_segments: int = 0
    processing_time: float = 0.0
    temp_dir: Optional[str] = None
    output_dirs: List[str] = None
    error_message: Optional[str] = None
    audio_duration: float = 0.0
    total_speech_time: float = 0.0
    speech_ratio: float = 0.0
    speaker_durations: Dict[str, float] = None

class LongAudioProcessor:
    """长音频处理器"""
    
    def __init__(self, config: LongAudioProcessingConfig):
        self.config = config
        
        # 设置日志
        self._setup_logging()
        
        # 初始化子模块
        try:
            self.speaker_diarizer = LongAudioSpeakerDiarizer(config)
            logger.info("成功初始化说话人分离器")
        except Exception as e:
            logger.error(f"初始化说话人分离器失败: {e}")
            raise
            
        try:
            self.quality_filter = LongAudioQualityFilter(config)
            logger.info("成功初始化质量筛选器")
        except Exception as e:
            logger.error(f"初始化质量筛选器失败: {e}")
            raise
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_speakers': 0,
            'total_segments': 0,
            'passed_segments': 0,
            'total_processing_time': 0.0,
            'total_audio_duration': 0.0,
            'total_speech_time': 0.0
        }
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(
            log_dir / self.config.log_file, 
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # 清除现有的处理器，避免重复添加
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def find_audio_files(self, input_dir: str) -> List[str]:
        """
        查找目录中的音频文件
        
        Args:
            input_dir: 输入目录
            
        Returns:
            音频文件路径列表
        """
        audio_files = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return audio_files
        
        for ext in self.config.processing.supported_formats:
            pattern = f"**/*{ext}"
            files = list(input_path.glob(pattern))
            audio_files.extend([str(f) for f in files])
        
        logger.info(f"在目录 {input_dir} 中找到 {len(audio_files)} 个音频文件")
        return audio_files
    
    def process_single_audio(self, audio_path: str) -> ProcessingResult:
        """
        处理单个长音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        audio_path_obj = Path(audio_path)
        audio_id = audio_path_obj.stem
        
        logger.info(f"开始处理长音频: {audio_path}")
        
        result = ProcessingResult(
            audio_path=audio_path,
            audio_id=audio_id,
            success=False,
            output_dirs=[]
        )
        
        try:
            # 创建临时目录
            temp_base = Path(self.config.processing.temp_dir)
            temp_dir = temp_base / f"temp_{audio_id}_{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            result.temp_dir = str(temp_dir)
            
            logger.info(f"创建临时目录: {temp_dir}")
            
            # 步骤1: 说话人分离和VAD
            logger.info("=== 步骤1: 说话人分离和VAD ===")
            diarization_result = self.speaker_diarizer.process_with_vad_and_diarization(audio_path)
            
            if not diarization_result['success']:
                result.error_message = f"说话人分离失败: {diarization_result['error_message']}"
                return result
            
            result.speaker_count = diarization_result.get('speaker_count', 0)
            result.audio_duration = diarization_result.get('audio_duration', 0.0)
            result.total_speech_time = diarization_result.get('total_speech_time', 0.0)
            result.speech_ratio = diarization_result.get('speech_ratio', 0.0)
            result.speaker_durations = diarization_result.get('speaker_durations', {})
            
            logger.info(f"检测到 {result.speaker_count} 个说话人")
            logger.info(f"音频时长: {result.audio_duration:.2f}s, 语音时长: {result.total_speech_time:.2f}s")
            
            # 步骤2: 提取说话人音频片段（使用内存处理）
            logger.info("=== 步骤2: 提取说话人音频片段 ===")
            speaker_segments = diarization_result['speaker_segments']
            speaker_audio_segments = self.speaker_diarizer.extract_speaker_audio_segments(
                audio_path, speaker_segments
            )
            
            if not speaker_audio_segments:
                result.error_message = "音频分割失败，没有生成音频片段"
                return result
            
            # 步骤3: 质量筛选和保存
            logger.info("=== 步骤3: 质量筛选和保存 ===")
            filter_result = self.quality_filter.process_speaker_audio_segments(
                speaker_audio_segments, audio_id, self.config.output_dir
            )
            
            result.total_segments = filter_result['total_segments']
            result.passed_segments = filter_result['total_passed']
            result.success = True
            
            # 收集输出目录
            for speaker_id, speaker_result in filter_result['speaker_results'].items():
                result.output_dirs.append(speaker_result['output_directory'])
            
            # 保存处理摘要
            self._save_processing_summary(result, diarization_result, filter_result)
            
            logger.info(f"长音频处理完成: {audio_path}")
            logger.info(f"总片段: {result.total_segments}, 通过筛选: {result.passed_segments} ({result.passed_segments/result.total_segments*100:.1f}%)")
            
        except Exception as e:
            logger.error(f"处理长音频时发生错误: {e}")
            result.error_message = str(e)
        
        finally:
            # 清理临时目录
            if result.temp_dir and Path(result.temp_dir).exists():
                try:
                    shutil.rmtree(result.temp_dir)
                    logger.debug(f"清理临时目录: {result.temp_dir}")
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {e}")
            
            result.processing_time = time.time() - start_time
            logger.info(f"处理耗时: {result.processing_time:.2f} 秒")
        
        return result
    
    def _save_processing_summary(self, result: ProcessingResult, 
                               diarization_result: Dict, filter_result: Dict):
        """保存处理摘要信息"""
        try:
            summary = {
                'audio_path': result.audio_path,
                'audio_id': result.audio_id,
                'success': result.success,
                'processing_time': result.processing_time,
                'audio_info': {
                    'duration': result.audio_duration,
                    'total_speech_time': result.total_speech_time,
                    'speech_ratio': result.speech_ratio
                },
                'speaker_info': {
                    'speaker_count': result.speaker_count,
                    'speaker_durations': result.speaker_durations
                },
                'processing_results': {
                    'total_segments': result.total_segments,
                    'passed_segments': result.passed_segments,
                    'pass_rate': result.passed_segments / result.total_segments if result.total_segments > 0 else 0.0
                },
                'output_directories': result.output_dirs,
                'vad_info': {
                    'vad_segments_count': len(diarization_result.get('vad_segments', [])),
                    'speaker_segments_count': len(diarization_result.get('speaker_segments', []))
                },
                'quality_filter_results': filter_result.get('speaker_results', {}),
                'processing_timestamp': self._get_timestamp()
            }
            
            # 保存到输出目录
            summary_dir = Path(self.config.output_dir) / result.audio_id
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = summary_dir / "processing_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            logger.info(f"保存处理摘要: {summary_path}")
            
        except Exception as e:
            logger.error(f"保存处理摘要失败: {e}")
    
    def process_directory(self) -> Dict[str, Any]:
        """
        处理整个目录的长音频文件
        
        Returns:
            处理结果统计
        """
        logger.info("开始批量处理长音频文件")
        logger.info(f"输入目录: {self.config.input_dir}")
        logger.info(f"输出目录: {self.config.output_dir}")
        
        # 查找音频文件
        audio_files = self.find_audio_files(self.config.input_dir)
        if not audio_files:
            logger.error("没有找到音频文件")
            return self.stats
        
        self.stats['total_files'] = len(audio_files)
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 处理文件
        results = []
        
        if self.config.processing.max_workers == 1:
            # 串行处理
            for audio_file in audio_files:
                result = self.process_single_audio(audio_file)
                results.append(result)
                self._update_stats(result)
        else:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.config.processing.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_audio, audio_file): audio_file 
                    for audio_file in audio_files
                }
                
                for future in as_completed(future_to_file):
                    audio_file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self._update_stats(result)
                    except Exception as e:
                        logger.error(f"处理文件 {audio_file} 时发生异常: {e}")
        
        # 生成最终报告
        self._generate_final_report(results)
        
        logger.info("批量处理完成")
        return self.stats
    
    def _update_stats(self, result: ProcessingResult):
        """更新统计信息"""
        self.stats['processed_files'] += 1
        
        if result.success:
            self.stats['successful_files'] += 1
            self.stats['total_speakers'] += result.speaker_count
            self.stats['total_segments'] += result.total_segments
            self.stats['passed_segments'] += result.passed_segments
            self.stats['total_audio_duration'] += result.audio_duration
            self.stats['total_speech_time'] += result.total_speech_time
        else:
            self.stats['failed_files'] += 1
        
        self.stats['total_processing_time'] += result.processing_time
        
        # 打印进度
        progress = self.stats['processed_files'] / self.stats['total_files'] * 100
        logger.info(f"处理进度: {self.stats['processed_files']}/{self.stats['total_files']} ({progress:.1f}%)")
    
    def _generate_final_report(self, results: List[ProcessingResult]):
        """生成最终处理报告"""
        try:
            # 计算统计数据
            avg_processing_time = self.stats['total_processing_time'] / self.stats['total_files'] if self.stats['total_files'] > 0 else 0
            success_rate = self.stats['successful_files'] / self.stats['total_files'] * 100 if self.stats['total_files'] > 0 else 0
            pass_rate = self.stats['passed_segments'] / self.stats['total_segments'] * 100 if self.stats['total_segments'] > 0 else 0
            avg_speakers = self.stats['total_speakers'] / self.stats['successful_files'] if self.stats['successful_files'] > 0 else 0
            speech_ratio = self.stats['total_speech_time'] / self.stats['total_audio_duration'] * 100 if self.stats['total_audio_duration'] > 0 else 0
            
            report = {
                'processing_summary': {
                    'total_files': self.stats['total_files'],
                    'successful_files': self.stats['successful_files'],
                    'failed_files': self.stats['failed_files'],
                    'success_rate': success_rate,
                    'total_speakers': self.stats['total_speakers'],
                    'avg_speakers_per_file': avg_speakers,
                    'total_segments': self.stats['total_segments'],
                    'passed_segments': self.stats['passed_segments'],
                    'pass_rate': pass_rate,
                    'total_processing_time': self.stats['total_processing_time'],
                    'average_processing_time': avg_processing_time,
                    'total_audio_duration': self.stats['total_audio_duration'],
                    'total_speech_time': self.stats['total_speech_time'],
                    'overall_speech_ratio': speech_ratio
                },
                'file_results': [
                    {
                        'audio_path': r.audio_path,
                        'audio_id': r.audio_id,
                        'success': r.success,
                        'speaker_count': r.speaker_count,
                        'total_segments': r.total_segments,  
                        'passed_segments': r.passed_segments,
                        'processing_time': r.processing_time,
                        'audio_duration': r.audio_duration,
                        'speech_ratio': r.speech_ratio,
                        'error_message': r.error_message
                    }
                    for r in results
                ],
                'config': self.config.to_dict(),
                'generation_timestamp': self._get_timestamp()
            }
            
            report_path = Path(self.config.output_dir) / "final_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"生成最终报告: {report_path}")
            
            # 打印摘要
            logger.info("=== 处理摘要 ===")
            logger.info(f"总文件数: {self.stats['total_files']}")
            logger.info(f"成功处理: {self.stats['successful_files']} ({success_rate:.1f}%)")
            logger.info(f"处理失败: {self.stats['failed_files']}")
            logger.info(f"总说话人数: {self.stats['total_speakers']} (平均 {avg_speakers:.1f}/文件)")
            logger.info(f"总音频片段: {self.stats['total_segments']}")
            logger.info(f"通过筛选: {self.stats['passed_segments']} ({pass_rate:.1f}%)")
            logger.info(f"总音频时长: {self.stats['total_audio_duration']:.1f}秒")
            logger.info(f"语音时长: {self.stats['total_speech_time']:.1f}秒 ({speech_ratio:.1f}%)")
            logger.info(f"总处理时间: {self.stats['total_processing_time']:.2f}秒 (平均 {avg_processing_time:.2f}秒/文件)")
            
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        import datetime
        return datetime.datetime.now().isoformat()


def main():
    """主函数"""
    # 创建默认配置
    config = LongAudioProcessingConfig()
    
    # 创建处理器并运行
    processor = LongAudioProcessor(config)
    stats = processor.process_directory()
    
    print("处理完成！")
    return stats

if __name__ == "__main__":
    main() 