"""
语音筛选Pipeline主模块
整合VAD检测、语音识别、音质评估等功能
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import shutil
from pathlib import Path

from config import PipelineConfig, Config
from vad_detector import VADDetector
from speech_recognizer import SpeechRecognizer
from audio_quality_assessor import AudioQualityAssessor

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """单个音频文件的处理结果"""
    file_path: str
    relative_path: str
    passed: bool
    vad_segments: List[Tuple[float, float]]
    transcription: Dict[str, Any]
    quality_scores: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0

class SpeechFilterPipeline:
    """语音筛选Pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.vad_detector = VADDetector(config)
        self.speech_recognizer = SpeechRecognizer(config)
        self.audio_quality_assessor = AudioQualityAssessor(config)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'vad_failed': 0,
            'transcription_failed': 0,
            'quality_failed': 0,
            'language_mismatch': 0,
            'total_processing_time': 0.0
        }
        
        # 处理结果
        self.results: List[ProcessingResult] = []
    
    def process_directory(self, input_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """
        处理整个目录
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径，如果为None则使用配置中的默认路径
            
        Returns:
            处理结果统计
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        logger.info(f"开始处理目录：{input_dir}")
        
        # 查找所有支持的音频文件
        audio_files = self._find_audio_files(input_dir)
        self.stats['total_files'] = len(audio_files)
        
        if not audio_files:
            logger.warning(f"在目录 {input_dir} 中没有找到支持的音频文件")
            return self.get_statistics()
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 并行处理音频文件
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self._process_single_file, file_path, input_dir, output_dir): file_path
                for file_path in audio_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self.stats['processed_files'] += 1
                    
                    if result.passed:
                        self.stats['passed_files'] += 1
                    else:
                        self.stats['failed_files'] += 1
                        if result.error_message:
                            if 'vad' in result.error_message.lower():
                                self.stats['vad_failed'] += 1
                            elif 'transcription' in result.error_message.lower():
                                self.stats['transcription_failed'] += 1
                            elif 'quality' in result.error_message.lower():
                                self.stats['quality_failed'] += 1
                            elif 'language' in result.error_message.lower():
                                self.stats['language_mismatch'] += 1
                    
                    self.stats['total_processing_time'] += result.processing_time
                    
                    # 打印进度
                    if self.stats['processed_files'] % 10 == 0:
                        logger.info(f"已处理 {self.stats['processed_files']}/{self.stats['total_files']} 个文件")
                
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
                    self.stats['failed_files'] += 1
        
        # 保存处理结果
        self._save_results(output_dir)
        
        logger.info(f"处理完成，共处理 {self.stats['processed_files']} 个文件，通过 {self.stats['passed_files']} 个")
        
        return self.get_statistics()
    
    def _find_audio_files(self, directory: str) -> List[str]:
        """查找目录中的所有音频文件"""
        audio_files = []
        directory = Path(directory)
        
        for ext in self.config.supported_formats:
            # 递归搜索音频文件
            pattern = f"**/*{ext}"
            files = directory.glob(pattern)
            audio_files.extend([str(f) for f in files if f.is_file()])
        
        return sorted(audio_files)
    
    def _process_single_file(self, file_path: str, input_dir: str, output_dir: str) -> ProcessingResult:
        """处理单个音频文件"""
        import time
        start_time = time.time()
        
        # 计算相对路径
        relative_path = os.path.relpath(file_path, input_dir)
        
        logger.debug(f"开始处理文件：{relative_path}")
        
        try:
            # 1. VAD检测
            vad_segments = self.vad_detector.detect_speech_segments(file_path)
            if not vad_segments:
                return ProcessingResult(
                    file_path=file_path,
                    relative_path=relative_path,
                    passed=False,
                    vad_segments=[],
                    transcription={},
                    quality_scores={},
                    error_message="VAD检测未发现语音段",
                    processing_time=time.time() - start_time
                )
            
            # 2. 语音识别
            transcription_result = self.speech_recognizer.transcribe_audio(file_path)
            if not self.speech_recognizer.is_valid_transcription(transcription_result):
                return ProcessingResult(
                    file_path=file_path,
                    relative_path=relative_path,
                    passed=False,
                    vad_segments=vad_segments,
                    transcription=transcription_result,
                    quality_scores={},
                    error_message="转录结果无效或不符合要求",
                    processing_time=time.time() - start_time
                )
            
            # 3. 音质评估
            quality_result = self.audio_quality_assessor.assess_audio_quality(file_path)
            if not self.audio_quality_assessor.is_high_quality(quality_result):
                return ProcessingResult(
                    file_path=file_path,
                    relative_path=relative_path,
                    passed=False,
                    vad_segments=vad_segments,
                    transcription=transcription_result,
                    quality_scores=quality_result,
                    error_message="音质评估不符合要求",
                    processing_time=time.time() - start_time
                )
            
            # 4. 文件通过所有检查，复制到输出目录
            output_file_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            shutil.copy2(file_path, output_file_path)
            
            logger.info(f"文件通过筛选：{relative_path}")
            
            return ProcessingResult(
                file_path=file_path,
                relative_path=relative_path,
                passed=True,
                vad_segments=vad_segments,
                transcription=transcription_result,
                quality_scores=quality_result,
                processing_time=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"处理文件 {relative_path} 时发生异常: {str(e)}")
            return ProcessingResult(
                file_path=file_path,
                relative_path=relative_path,
                passed=False,
                vad_segments=[],
                transcription={},
                quality_scores={},
                error_message=f"处理异常: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _save_results(self, output_dir: str):
        """保存处理结果"""
        # 准备保存的数据
        results_data = {
            'config': {
                'vad_threshold': self.config.vad.threshold,
                'vad_hop_size': self.config.vad.hop_size,
                'vad_min_speech_duration': self.config.vad.min_speech_duration,
                'vad_max_speech_duration': self.config.vad.max_speech_duration,
                'whisper_model': self.config.asr.model_name,
                'whisper_language': self.config.asr.language,
                'distilmos_threshold': self.config.audio_quality.distil_mos_threshold,
                'dnsmos_threshold': self.config.audio_quality.dnsmos_threshold,
                'supported_formats': self.config.processing.supported_formats
            },
            'statistics': self.stats,
            'results': []
        }
        
        # 转换处理结果为可序列化格式
        for result in self.results:
            result_dict = {
                'file_path': result.file_path,
                'relative_path': result.relative_path,
                'passed': result.passed,
                'vad_segments': result.vad_segments,
                'transcription': result.transcription,
                'quality_scores': result.quality_scores,
                'error_message': result.error_message,
                'processing_time': result.processing_time
            }
            results_data['results'].append(result_dict)
        
        # 保存到JSON文件
        results_file = os.path.join(output_dir, self.config.results_file)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"处理结果已保存到：{results_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = self.stats.copy()
        
        # 计算百分比
        if stats['total_files'] > 0:
            stats['pass_rate'] = (stats['passed_files'] / stats['total_files']) * 100
            stats['fail_rate'] = (stats['failed_files'] / stats['total_files']) * 100
        else:
            stats['pass_rate'] = 0.0
            stats['fail_rate'] = 0.0
        
        # 计算平均处理时间
        if stats['processed_files'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['processed_files']
        else:
            stats['avg_processing_time'] = 0.0
        
        return stats
    
    def print_statistics(self):
        """打印处理统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("          语音筛选处理统计")
        print("="*50)
        print(f"总文件数:           {stats['total_files']}")
        print(f"已处理文件数:       {stats['processed_files']}")
        print(f"通过筛选文件数:     {stats['passed_files']}")
        print(f"未通过筛选文件数:   {stats['failed_files']}")
        print(f"通过率:             {stats['pass_rate']:.2f}%")
        print(f"失败率:             {stats['fail_rate']:.2f}%")
        print()
        print("失败原因统计:")
        print(f"  VAD检测失败:      {stats['vad_failed']}")
        print(f"  转录失败:         {stats['transcription_failed']}")
        print(f"  音质评估失败:     {stats['quality_failed']}")
        print(f"  语言不匹配:       {stats['language_mismatch']}")
        print()
        print(f"总处理时间:         {stats['total_processing_time']:.2f}秒")
        print(f"平均处理时间:       {stats['avg_processing_time']:.2f}秒/文件")
        print("="*50)
    
    def get_passed_files(self) -> List[str]:
        """获取通过筛选的文件列表"""
        return [result.relative_path for result in self.results if result.passed]
    
    def get_failed_files(self) -> List[Tuple[str, str]]:
        """获取未通过筛选的文件列表及失败原因"""
        return [(result.relative_path, result.error_message or "未知错误") 
                for result in self.results if not result.passed]
    
    def export_transcriptions(self, output_dir: str):
        """导出转录文本"""
        transcriptions = []
        
        for result in self.results:
            if result.passed and result.transcription.get('text'):
                transcriptions.append({
                    'file_path': result.relative_path,
                    'text': result.transcription['text'],
                    'language': result.transcription.get('language', 'unknown'),
                    'word_count': result.transcription.get('word_count', 0)
                })
        
        # 保存转录文本
        transcription_file = os.path.join(output_dir, 'transcriptions.json')
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"转录文本已保存到：{transcription_file}")
    
    def export_quality_report(self, output_dir: str):
        """导出音质评估报告"""
        quality_data = []
        
        for result in self.results:
            if result.quality_scores and result.quality_scores.get('scores'):
                scores = result.quality_scores['scores']
                quality_data.append({
                    'file_path': result.relative_path,
                    'passed': result.passed,
                    'distilmos': scores.get('distilmos', 0),
                    'dnsmos': scores.get('dnsmos', 0),
                    'overall': scores.get('overall', 0)
                })
        
        # 保存音质报告
        quality_file = os.path.join(output_dir, 'quality_report.json')
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"音质报告已保存到：{quality_file}") 