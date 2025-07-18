# 语音筛选Pipeline - 多GPU并行处理版本
"""
多GPU并行处理版本的语音筛选Pipeline
支持将处理任务分发到多个GPU上同时处理，提高处理效率
"""

import os
import json
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

from config import PipelineConfig, Config
from pipeline import SpeechFilterPipeline
from vad_detector import VADDetector
from speech_recognizer import SpeechRecognizer
from audio_quality_assessor import AudioQualityAssessor

logger = logging.getLogger(__name__)

class MultiGPUPipeline:
    """多GPU并行处理Pipeline"""
    
    def __init__(self, config: PipelineConfig, num_gpus: int = 4, skip_processed: bool = False):
        self.config = config
        self.num_gpus = num_gpus
        self.skip_processed = skip_processed
        
        # 设置多进程启动方式
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # 验证GPU可用性
        self._validate_gpus()
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'passed_files': 0,
            'failed_files': 0,
            'gpu_stats': {i: {'processed': 0, 'skipped': 0, 'passed': 0, 'failed': 0} for i in range(num_gpus)},
            'total_processing_time': 0.0
        }
        
        # 存储所有结果
        self.all_results = []
    
    def _validate_gpus(self):
        """验证GPU可用性"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU处理")
            return
        
        available_gpus = torch.cuda.device_count()
        if available_gpus < self.num_gpus:
            logger.warning(f"可用GPU数量({available_gpus})少于请求数量({self.num_gpus})，将使用{available_gpus}个GPU")
            self.num_gpus = available_gpus
        
        for i in range(self.num_gpus):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                logger.info(f"GPU {i} 验证通过")
            except Exception as e:
                logger.error(f"GPU {i} 验证失败: {e}")
                raise
    
    def _find_audio_files(self, directory: str) -> List[str]:
        """查找目录中的所有音频文件"""
        audio_files = []
        directory = Path(directory)
        
        for ext in self.config.supported_formats:
            pattern = f"**/*{ext}"
            files = directory.glob(pattern)
            audio_files.extend([str(f) for f in files if f.is_file()])
        
        return sorted(audio_files)
    
    def _is_file_processed(self, file_path: str, input_dir: str, output_dir: str) -> bool:
        """检查文件是否已经处理过"""
        try:
            # 计算相对路径
            relative_path = os.path.relpath(file_path, input_dir)
            
            # 计算JSON文件路径
            audio_filename = os.path.basename(relative_path)
            audio_dirname = os.path.dirname(relative_path)
            json_filename = f"{audio_filename}.json"
            json_file_path = os.path.join(output_dir, audio_dirname, json_filename)
            
            # 检查JSON文件是否存在
            if not os.path.exists(json_file_path):
                return False
            
            # 读取JSON文件并检查内容
            with open(json_file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 检查是否包含完整的处理结果
            required_fields = ['file_path', 'passed', 'vad_segments', 'transcription', 'quality_scores']
            for field in required_fields:
                if field not in result_data:
                    return False
            
            # 检查是否有处理时间戳
            if 'timestamp' not in result_data:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"检查文件处理状态失败 {file_path}: {e}")
            return False
    
    def _filter_processed_files(self, audio_files: List[str], input_dir: str, output_dir: str) -> Tuple[List[str], List[str]]:
        """过滤已处理的文件，返回需要处理的文件列表和跳过的文件列表"""
        if not self.skip_processed:
            return audio_files, []
        
        files_to_process = []
        skipped_files = []
        
        logger.info("正在检查已处理文件...")
        
        for file_path in audio_files:
            if self._is_file_processed(file_path, input_dir, output_dir):
                skipped_files.append(file_path)
            else:
                files_to_process.append(file_path)
        
        logger.info(f"文件处理状态检查完成: {len(files_to_process)} 个需要处理, {len(skipped_files)} 个已跳过")
        
        return files_to_process, skipped_files
    
    def _split_files_to_gpus(self, audio_files: List[str]) -> List[List[str]]:
        """将音频文件分割到各个GPU"""
        if not audio_files:
            return [[] for _ in range(self.num_gpus)]
        
        chunk_size = len(audio_files) // self.num_gpus
        chunks = []
        
        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            if i == self.num_gpus - 1:
                end_idx = len(audio_files)
            else:
                end_idx = (i + 1) * chunk_size
            
            chunks.append(audio_files[start_idx:end_idx])
        
        for i, chunk in enumerate(chunks):
            logger.info(f"GPU {i}: {len(chunk)} 个文件")
        
        return chunks
    
    def process_directory(self, input_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """
        处理整个目录，使用多GPU并行处理
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            
        Returns:
            处理结果统计
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        logger.info(f"开始多GPU并行处理，使用{self.num_gpus}个GPU")
        logger.info(f"输入目录: {input_dir}")
        logger.info(f"输出目录: {output_dir}")
        
        # 查找所有音频文件
        audio_files = self._find_audio_files(input_dir)
        self.stats['total_files'] = len(audio_files)
        
        if not audio_files:
            logger.warning(f"在目录 {input_dir} 中没有找到支持的音频文件")
            return self.stats
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建日志目录 - 放在上级目录的logs下
        log_dir = os.path.join(os.path.dirname(output_dir), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 过滤已处理的文件
        files_to_process, skipped_files = self._filter_processed_files(audio_files, input_dir, output_dir)
        self.stats['skipped_files'] = len(skipped_files)
        
        if not files_to_process:
            logger.info("所有文件都已处理完成，无需重新处理")
            return self.stats
        
        logger.info(f"需要处理 {len(files_to_process)} 个文件，跳过 {len(skipped_files)} 个已处理文件")
        
        # 将文件分割到各个GPU
        gpu_file_chunks = self._split_files_to_gpus(files_to_process)
        
        # 使用多进程处理各个GPU
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            future_to_gpu = {}
            for gpu_id, file_chunk in enumerate(gpu_file_chunks):
                if file_chunk:
                    future = executor.submit(
                        self._process_on_gpu,
                        gpu_id,
                        file_chunk,
                        input_dir,
                        output_dir,
                        log_dir
                    )
                    future_to_gpu[future] = gpu_id
            
            # 收集结果
            for future in as_completed(future_to_gpu):
                gpu_id = future_to_gpu[future]
                try:
                    gpu_stats, gpu_results = future.result()
                    self.stats['gpu_stats'][gpu_id] = gpu_stats
                    self.stats['processed_files'] += gpu_stats['processed']
                    self.stats['passed_files'] += gpu_stats['passed']
                    self.stats['failed_files'] += gpu_stats['failed']
                    self.all_results.extend(gpu_results)
                    
                    logger.info(f"GPU {gpu_id} 处理完成: {gpu_stats['processed']} 个文件，{gpu_stats['passed']} 个通过")
                    
                except Exception as e:
                    logger.error(f"GPU {gpu_id} 处理失败: {e}")
                    self.stats['failed_files'] += len(gpu_file_chunks[gpu_id])
        
        self.stats['total_processing_time'] = time.time() - start_time
        
        # 保存统计结果
        self._save_multi_gpu_stats(output_dir)
        
        # 保存详细结果索引
        self.save_detailed_results(output_dir)
        
        logger.info(f"多GPU处理完成，总耗时: {self.stats['total_processing_time']:.2f}秒")
        logger.info(f"处理文件: {self.stats['processed_files']}/{self.stats['total_files']}")
        logger.info(f"通过文件: {self.stats['passed_files']}/{self.stats['processed_files']}")
        
        return self.stats
    
    def _process_on_gpu(self, gpu_id: int, file_chunk: List[str], input_dir: str, output_dir: str, log_dir: str) -> tuple:
        """
        在指定GPU上处理文件块
        
        Args:
            gpu_id: GPU ID
            file_chunk: 文件列表
            input_dir: 输入目录
            output_dir: 输出目录
            log_dir: 日志目录
            
        Returns:
            (GPU处理统计, 处理结果列表)
        """
        # 设置GPU设备环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 重新初始化CUDA环境
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 设置GPU专用日志
        gpu_log_file = os.path.join(log_dir, f"gpu_{gpu_id}_processing.log")
        gpu_logger = logging.getLogger(f"gpu_{gpu_id}")
        gpu_logger.setLevel(logging.INFO)
        
        # 创建GPU专用日志处理器
        gpu_handler = logging.FileHandler(gpu_log_file, encoding='utf-8')
        gpu_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        gpu_handler.setFormatter(formatter)
        
        # 避免重复添加handler
        if not gpu_logger.handlers:
            gpu_logger.addHandler(gpu_handler)
        
        gpu_logger.info(f"GPU {gpu_id} 开始处理 {len(file_chunk)} 个文件")
        
        # 在子进程中，设备总是映射到cuda:0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 创建GPU专用配置
        gpu_config = self._create_gpu_config(device)
        
        # 创建单GPU pipeline
        pipeline = SpeechFilterPipeline(gpu_config)
        
        # 处理文件
        gpu_stats = {'processed': 0, 'passed': 0, 'failed': 0}
        gpu_results = []
        
        for file_path in file_chunk:
            try:
                # 处理单个文件
                result = pipeline._process_single_file(file_path, input_dir, output_dir)
                gpu_results.append(result)
                
                # 实时保存单个音频的详细结果
                self._save_single_audio_result(result, gpu_id, input_dir, output_dir, gpu_logger)
                
                gpu_stats['processed'] += 1
                if result.passed:
                    gpu_stats['passed'] += 1
                else:
                    gpu_stats['failed'] += 1
                
                # 定期报告进度
                if gpu_stats['processed'] % 100 == 0:
                    gpu_logger.info(f"GPU {gpu_id} 已处理 {gpu_stats['processed']}/{len(file_chunk)} 个文件")
                    logger.info(f"GPU {gpu_id} 已处理 {gpu_stats['processed']}/{len(file_chunk)} 个文件")
                
            except Exception as e:
                gpu_logger.error(f"GPU {gpu_id} 处理文件 {file_path} 失败: {e}")
                logger.error(f"GPU {gpu_id} 处理文件 {file_path} 失败: {e}")
                gpu_stats['failed'] += 1
        
        gpu_logger.info(f"GPU {gpu_id} 处理完成: {gpu_stats}")
        logger.info(f"GPU {gpu_id} 处理完成: {gpu_stats}")
        
        return gpu_stats, gpu_results
    
    def _save_single_audio_result(self, result, gpu_id: int, input_dir: str, output_dir: str, gpu_logger):
        """
        实时保存单个音频的详细结果到输出目录中与音频文件相同的位置
        
        Args:
            result: 处理结果
            gpu_id: GPU ID
            input_dir: 输入目录
            output_dir: 输出目录
            gpu_logger: GPU专用日志记录器
        """
        try:
            # 将ProcessingResult转换为字典
            result_dict = {
                'file_path': result.file_path,
                'relative_path': result.relative_path,
                'passed': result.passed,
                'vad_segments': self._convert_to_json_serializable(result.vad_segments),
                'transcription': self._convert_to_json_serializable(result.transcription),
                'quality_scores': self._convert_to_json_serializable(result.quality_scores),
                'error_message': result.error_message,
                'processing_time': float(result.processing_time),
                'gpu_id': gpu_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 计算JSON文件的保存路径
            relative_path = result.relative_path
            audio_filename = os.path.basename(relative_path)
            audio_dirname = os.path.dirname(relative_path)
            
            json_filename = f"{audio_filename}.json"
            json_dir = os.path.join(output_dir, audio_dirname)
            os.makedirs(json_dir, exist_ok=True)
            json_file_path = os.path.join(json_dir, json_filename)
            
            # 保存详细结果
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            gpu_logger.debug(f"GPU {gpu_id} 保存音频详细结果: {json_file_path}")
            
        except Exception as e:
            gpu_logger.error(f"GPU {gpu_id} 保存音频详细结果失败 {result.relative_path}: {e}")
    
    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        else:
            return obj
    
    def _create_gpu_config(self, device: str) -> PipelineConfig:
        """为特定GPU创建配置"""
        gpu_config = PipelineConfig()
        
        # 复制原始配置
        gpu_config.vad = self.config.vad
        gpu_config.asr = self.config.asr
        gpu_config.audio_quality = self.config.audio_quality
        gpu_config.processing = self.config.processing
        
        # 设置GPU设备
        gpu_config.asr.device = device
        
        # 减少并行数以适应GPU内存
        gpu_config.num_workers = 1
        
        return gpu_config
    
    def _save_multi_gpu_stats(self, output_dir: str):
        """保存多GPU处理统计"""
        # 统计文件保存在上级目录
        stats_file = os.path.join(os.path.dirname(output_dir), "results", "multi_gpu_stats.json")
        
        # 添加时间戳
        self.stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 计算百分比
        if self.stats['total_files'] > 0:
            self.stats['pass_rate'] = (self.stats['passed_files'] / self.stats['total_files']) * 100
            self.stats['fail_rate'] = (self.stats['failed_files'] / self.stats['total_files']) * 100
        else:
            self.stats['pass_rate'] = 0.0
            self.stats['fail_rate'] = 0.0
        
        # 保存统计
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"多GPU统计已保存: {stats_file}")
    
    def export_transcriptions(self, output_dir: str):
        """导出转录文本汇总"""
        transcriptions = []
        
        for result in self.all_results:
            if result.passed and result.transcription.get('text'):
                transcriptions.append({
                    'file_path': result.relative_path,
                    'text': result.transcription['text'],
                    'language': result.transcription.get('language', 'unknown'),
                    'word_count': result.transcription.get('word_count', 0)
                })
        
        # 保存汇总的转录文本 - 保存在上级目录
        transcription_file = os.path.join(os.path.dirname(output_dir), 'results', 'multi_gpu_transcriptions.json')
        with open(transcription_file, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"转录文本汇总已保存到：{transcription_file}")
    
    def export_quality_report(self, output_dir: str):
        """导出音质评估报告汇总"""
        quality_data = []
        
        for result in self.all_results:
            if result.quality_scores and result.quality_scores.get('scores'):
                scores = result.quality_scores['scores']
                quality_data.append({
                    'file_path': result.relative_path,
                    'passed': result.passed,
                    'distilmos': scores.get('distilmos', 0),
                    'dnsmos': scores.get('dnsmos', 0),
                    'dnsmospro': scores.get('dnsmospro', 0),
                    'overall': scores.get('overall', 0)
                })
        
        # 保存汇总的音质报告 - 保存在上级目录
        quality_file = os.path.join(os.path.dirname(output_dir), 'results', 'multi_gpu_quality_report.json')
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"音质报告汇总已保存到：{quality_file}")
    
    def save_detailed_results(self, output_dir: str):
        """保存每条音频的详细结果索引"""
        # 统计已保存的详细结果文件
        json_count = 0
        processed_files = []
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.json') and not file.startswith('multi_gpu_'):
                    json_count += 1
                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                    processed_files.append(rel_path)
        
        # 创建详细结果索引文件
        index_data = {
            'total_json_files': json_count,
            'processed_files': self.stats['processed_files'],
            'skipped_files': self.stats['skipped_files'],
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_count': self.num_gpus,
            'description': '每条音频的详细处理结果JSON文件已保存在与音频文件相同的目录中',
            'note': 'JSON文件包含VAD、识别和音质评估信息，与对应的音频文件在同一目录',
            'processed_files_list': sorted(processed_files)
        }
        
        # 如果启用了跳过处理，添加跳过文件信息
        if self.skip_processed:
            index_data['skip_processed_enabled'] = True
            index_data['note'] += '，跳过了已处理的文件'
        
        # 索引文件保存在上级目录
        index_file = os.path.join(os.path.dirname(output_dir), 'results', 'detailed_results_index.json')
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细结果索引已保存: {index_file}")
    
    def print_statistics(self):
        """打印处理统计信息"""
        stats = self.stats
        
        print("\n" + "="*60)
        print("          多GPU语音筛选处理统计")
        print("="*60)
        print(f"总文件数:           {stats['total_files']}")
        print(f"已处理文件数:       {stats['processed_files']}")
        if self.skip_processed:
            print(f"跳过文件数:         {stats['skipped_files']}")
        print(f"通过筛选文件数:     {stats['passed_files']}")
        print(f"未通过筛选文件数:   {stats['failed_files']}")
        print(f"通过率:             {stats.get('pass_rate', 0):.2f}%")
        print(f"失败率:             {stats.get('fail_rate', 0):.2f}%")
        print()
        print("GPU处理统计:")
        for gpu_id, gpu_stat in stats['gpu_stats'].items():
            print(f"  GPU {gpu_id}: 处理{gpu_stat['processed']}，通过{gpu_stat['passed']}，失败{gpu_stat['failed']}")
        print()
        print(f"总处理时间:         {stats['total_processing_time']:.2f}秒")
        if stats['processed_files'] > 0:
            avg_time = stats['total_processing_time'] / stats['processed_files']
            print(f"平均处理时间:       {avg_time:.2f}秒/文件")
        if self.skip_processed and stats['skipped_files'] > 0:
            print(f"跳过文件节省时间:   预计节省{stats['skipped_files'] * avg_time:.2f}秒" if stats['processed_files'] > 0 else "")
        print("="*60) 