"""
多GPU并行长音频处理器
充分利用多张GPU进行并行音频处理
"""
import os
import sys
import time
import json
import fcntl
import logging
import threading
import multiprocessing as mp
from multiprocessing import Process, Pool
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from config import LongAudioProcessingConfig

# 设置multiprocessing启动方法为spawn以支持CUDA
if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass


@dataclass
class MultiGPUConfig:
    """多GPU配置"""
    num_gpus: int = -1  # -1表示使用所有GPU
    gpu_memory_fraction: float = 0.9  # 每个GPU使用的显存比例
    max_concurrent_files: int = 8  # 最大并发处理文件数
    load_balance_strategy: str = "round_robin"  # round_robin, memory_based
    enable_gpu_monitoring: bool = True
    

class SimpleGPUManager:
    """简化的GPU资源管理器 - 使用文件锁避免多进程问题"""
    
    def __init__(self, config: MultiGPUConfig, work_dir: str = "gpu_locks"):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 延迟GPU检测，避免在主进程中初始化CUDA
        self.num_gpus = None
        
        print(f"SimpleGPU管理器初始化完成，工作目录: {work_dir}")
    
    def _detect_gpus(self) -> int:
        """检测可用GPU数量（延迟初始化）"""
        if self.num_gpus is None:
            # 在子进程中才检测GPU，避免主进程初始化CUDA
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA不可用，无法使用GPU加速")
                
                num_gpus = torch.cuda.device_count()
                if self.config.num_gpus == -1:
                    self.num_gpus = num_gpus
                else:
                    self.num_gpus = min(self.config.num_gpus, num_gpus)
                    
                # 初始化GPU锁文件
                for gpu_id in range(self.num_gpus):
                    lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
                    if not lock_file.exists():
                        lock_file.write_text(json.dumps({
                            'gpu_id': gpu_id,
                            'in_use': False,
                            'current_file': None,
                            'processed_count': 0,
                            'last_update': time.time()
                        }))
            except Exception as e:
                print(f"GPU检测失败: {e}")
                self.num_gpus = 0
                
        return self.num_gpus
    
    def acquire_gpu(self, timeout: float = 30.0) -> Optional[int]:
        """获取可用GPU"""
        num_gpus = self._detect_gpus()
        if num_gpus == 0:
            return None
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for gpu_id in range(num_gpus):
                lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
                
                try:
                    with open(lock_file, 'r+') as f:
                        # 尝试获取文件锁
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        # 读取当前状态
                        f.seek(0)
                        data = json.load(f)
                        
                        if not data['in_use']:
                            # 标记为使用中
                            data['in_use'] = True
                            data['last_update'] = time.time()
                            
                            # 写回文件
                            f.seek(0)
                            f.truncate()
                            json.dump(data, f)
                            f.flush()
                            
                            # 释放锁
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            return gpu_id
                        
                        # 释放锁
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
                except (IOError, OSError, json.JSONDecodeError):
                    # 文件被占用或损坏，跳过
                    continue
            
            time.sleep(0.1)  # 短暂等待后重试
        
        return None  # 超时未获取到GPU
    
    def release_gpu(self, gpu_id: int):
        """释放GPU"""
        lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
        
        try:
            with open(lock_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                # 读取当前状态
                f.seek(0)
                data = json.load(f)
                
                # 更新状态
                data['in_use'] = False
                data['current_file'] = None
                data['processed_count'] += 1
                data['last_update'] = time.time()
                
                # 写回文件
                f.seek(0)
                f.truncate()
                json.dump(data, f)
                f.flush()
                
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
        except (IOError, json.JSONDecodeError) as e:
            print(f"释放GPU {gpu_id} 时出错: {e}")
    
    def update_gpu_status(self, gpu_id: int, file_path: str):
        """更新GPU处理状态"""
        lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
        
        try:
            with open(lock_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                f.seek(0)
                data = json.load(f)
                data['current_file'] = str(file_path)
                data['last_update'] = time.time()
                
                f.seek(0)
                f.truncate()
                json.dump(data, f)
                f.flush()
                
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
        except (IOError, json.JSONDecodeError):
            pass  # 更新失败不影响处理
    
    def get_stats(self) -> Dict[int, Dict]:
        """获取GPU统计信息"""
        num_gpus = self._detect_gpus()
        stats = {}
        
        for gpu_id in range(num_gpus):
            lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
            
            try:
                with open(lock_file, 'r') as f:
                    data = json.load(f)
                    stats[gpu_id] = data
            except (IOError, json.JSONDecodeError):
                stats[gpu_id] = {
                    'gpu_id': gpu_id,
                    'in_use': False,
                    'current_file': None,
                    'processed_count': 0,
                    'last_update': 0
                }
        
        return stats


def process_single_file_multiprocess(args: Tuple[str, LongAudioProcessingConfig, int, str]) -> Dict[str, Any]:
    """多进程处理单个文件的工作函数"""
    file_path, base_config, process_id, work_dir = args
    
    import copy
    import time
    import torch
    
    # 在子进程中初始化CUDA相关模块
    try:
        from long_audio_processor import LongAudioProcessor
    except ImportError as e:
        return {
            'file_path': str(file_path),
            'success': False,
            'gpu_id': None,
            'process_id': process_id,
            'processing_time': 0,
            'error_message': f'导入LongAudioProcessor失败: {e}'
        }
    
    # 创建进程内的GPU管理器
    gpu_manager = SimpleGPUManager(MultiGPUConfig(), work_dir)
    
    logger = logging.getLogger(f"MultiGPU-Process-{process_id}")
    
    start_time = time.time()
    gpu_id = None
    
    try:
        # 获取GPU资源
        gpu_id = gpu_manager.acquire_gpu()
        if gpu_id is None:
            return {
                'file_path': str(file_path),
                'success': False,
                'gpu_id': None,
                'process_id': process_id,
                'processing_time': time.time() - start_time,
                'error_message': 'GPU资源获取超时'
            }
        
        logger.info(f"进程 {process_id} 获取到 GPU {gpu_id}，开始处理: {Path(file_path).name}")
        
        # 更新GPU状态
        gpu_manager.update_gpu_status(gpu_id, str(file_path))
        
        # 在子进程中设置CUDA设备
        torch.cuda.set_device(gpu_id)
        
        # 创建专用于此GPU的配置
        gpu_config = copy.deepcopy(base_config)
        gpu_config.whisper.device = f"cuda:{gpu_id}"
        gpu_config.processing.temp_dir = f"temp_gpu_{gpu_id}_process_{process_id}"
        gpu_config._gpu_device = f"cuda:{gpu_id}"
        gpu_config._process_id = process_id  # 添加进程ID用于文件命名
        
        # 创建处理器实例
        processor = LongAudioProcessor(gpu_config)
        
        # 处理文件
        result = processor.process_single_audio(str(file_path))
        
        processing_time = time.time() - start_time
        
        logger.info(f"进程 {process_id} 在 GPU {gpu_id} 上完成处理: {Path(file_path).name} (耗时: {processing_time:.2f}s)")
        
        return {
            'file_path': str(file_path),
            'success': result.success,
            'gpu_id': gpu_id,
            'process_id': process_id,
            'processing_time': processing_time,
            'total_segments': result.total_segments,
            'passed_segments': result.passed_segments,
            'output_dirs': result.output_dirs,
            'error_message': result.error_message if not result.success else None
        }
        
    except Exception as e:
        logger.error(f"进程 {process_id} 在 GPU {gpu_id} 处理文件 {file_path} 失败: {e}")
        return {
            'file_path': str(file_path),
            'success': False,
            'gpu_id': gpu_id,
            'process_id': process_id,
            'processing_time': time.time() - start_time,
            'error_message': str(e)
        }
    
    finally:
        # 释放GPU资源
        if gpu_id is not None:
            gpu_manager.release_gpu(gpu_id)
            logger.debug(f"进程 {process_id} 释放 GPU {gpu_id}")


class MultiGPULongAudioProcessor:
    """多GPU长音频处理器"""
    
    def __init__(self, base_config: LongAudioProcessingConfig, multi_gpu_config: MultiGPUConfig = None):
        self.base_config = base_config
        self.multi_gpu_config = multi_gpu_config or MultiGPUConfig()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建工作目录
        self.work_dir = Path("gpu_work") / f"session_{int(time.time())}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化GPU管理器（不在主进程中检测GPU数量）
        self.gpu_manager = SimpleGPUManager(self.multi_gpu_config, str(self.work_dir))
        
        # 通过临时检测来显示GPU信息（但不保持CUDA上下文）
        self._show_gpu_info()
    
    def _show_gpu_info(self):
        """显示GPU信息但不保持CUDA上下文"""
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if self.multi_gpu_config.num_gpus != -1:
                    num_gpus = min(self.multi_gpu_config.num_gpus, num_gpus)
                
                self.logger.info(f"初始化多GPU处理器，检测到 {num_gpus} 张GPU")
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    self.logger.info(f"GPU {i}: {props.name} - {props.total_memory/1024**3:.1f}GB")
            else:
                self.logger.warning("CUDA不可用")
        except Exception as e:
            self.logger.warning(f"GPU信息获取失败: {e}")
    
    def process_directory_parallel(self) -> Dict[str, Any]:
        """并行处理目录中的所有音频文件"""
        # 查找音频文件
        audio_files = self._find_audio_files()
        total_files = len(audio_files)
        
        if total_files == 0:
            self.logger.warning(f"在目录 {self.base_config.input_dir} 中未找到音频文件")
            return {'total_files': 0, 'successful_files': 0, 'failed_files': 0, 'results': []}
        
        self.logger.info(f"找到 {total_files} 个音频文件，开始多GPU并行处理")
        
        # 创建输出目录
        Path(self.base_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 准备进程池参数 - 基于配置估算合理的进程数
        estimated_gpus = self.multi_gpu_config.num_gpus
        if estimated_gpus == -1:
            try:
                import torch
                estimated_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except:
                estimated_gpus = 1
        
        max_processes = min(
            self.multi_gpu_config.max_concurrent_files,
            estimated_gpus,
            total_files
        )
        
        self.logger.info(f"使用 {max_processes} 个并行进程处理")
        
        # 准备任务参数
        task_args = [
            (str(file_path), self.base_config, i % max_processes, str(self.work_dir))
            for i, file_path in enumerate(audio_files)
        ]
        
        successful_files = []
        failed_files = []
        start_time = time.time()
        
        # 启动进度监控线程
        progress_stop = threading.Event()
        progress_thread = threading.Thread(
            target=self._monitor_progress_thread,
            args=(progress_stop, total_files, start_time)
        )
        progress_thread.start()
        
        try:
            # 使用spawn方式的进程池进行并行处理
            with Pool(processes=max_processes) as pool:
                
                self.logger.info("开始并行处理...")
                
                # 批处理以避免内存问题
                batch_size = max_processes * 2
                for i in range(0, len(task_args), batch_size):
                    batch_args = task_args[i:i + batch_size]
                    batch_results = pool.map(process_single_file_multiprocess, batch_args)
                    
                    for result in batch_results:
                        if result['success']:
                            successful_files.append(result)
                        else:
                            failed_files.append(result)
                    
                    self.logger.info(f"完成批次 {i//batch_size + 1}/{(len(task_args) + batch_size - 1)//batch_size}")
        
        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"多进程处理过程中发生错误: {e}")
        
        finally:
            # 停止进度监控
            progress_stop.set()
            progress_thread.join()
            
            # 清理工作目录
            try:
                import shutil
                shutil.rmtree(self.work_dir)
            except:
                pass
        
        total_time = time.time() - start_time
        
        # 统计最终结果
        final_stats = {
            'total_files': total_files,
            'successful_files': len(successful_files),
            'failed_files': len(failed_files),
            'success_rate': len(successful_files) / total_files * 100 if total_files > 0 else 0,
            'gpu_stats': self.gpu_manager.get_stats(),
            'processing_time': total_time,
            'average_time_per_file': total_time / total_files if total_files > 0 else 0,
            'successful_results': successful_files,
            'failed_results': failed_files
        }
        
        self.logger.info(f"多GPU处理完成: {final_stats['successful_files']}/{total_files} 文件成功处理 "
                         f"({final_stats['success_rate']:.1f}%) - 总耗时: {total_time/60:.1f}分钟")
        
        # 输出GPU使用统计
        self.logger.info("=== GPU使用统计 ===")
        for gpu_id, stats in final_stats['gpu_stats'].items():
            self.logger.info(f"GPU {gpu_id}: 处理了 {stats['processed_count']} 个文件")
        
        return final_stats
    
    def _monitor_progress_thread(self, stop_event: threading.Event, total_files: int, start_time: float):
        """进度监控线程"""
        while not stop_event.is_set():
            if stop_event.wait(10):  # 每10秒检查一次
                break
            
            try:
                gpu_stats = self.gpu_manager.get_stats()
                total_processed = sum(stats['processed_count'] for stats in gpu_stats.values())
                
                if total_processed > 0:
                    elapsed_time = time.time() - start_time
                    progress_percent = (total_processed / total_files) * 100
                    estimated_total_time = elapsed_time * total_files / total_processed
                    remaining_time = estimated_total_time - elapsed_time
                    
                    active_gpus = sum(1 for stats in gpu_stats.values() if stats['in_use'])
                    
                    self.logger.info(
                        f"处理进度: {total_processed}/{total_files} ({progress_percent:.1f}%) "
                        f"- 已用时: {elapsed_time/60:.1f}分钟, 预计剩余: {remaining_time/60:.1f}分钟 "
                        f"- 活跃GPU: {active_gpus}/{len(gpu_stats)}"
                    )
            except Exception as e:
                self.logger.error(f"进度监控出错: {e}")
    
    def _find_audio_files(self) -> List[Path]:
        """查找音频文件"""
        input_path = Path(self.base_config.input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_path}")
        
        audio_files = []
        for ext in self.base_config.processing.supported_formats:
            audio_files.extend(input_path.rglob(f"*{ext}"))
            audio_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        return sorted(audio_files)


def main():
    """主函数，用于测试多GPU处理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多GPU长音频处理器")
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--num-gpus', type=int, default=-1, help='使用的GPU数量（-1表示全部）')
    parser.add_argument('--max-concurrent', type=int, default=8, help='最大并发文件数')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    base_config = LongAudioProcessingConfig()
    base_config.input_dir = args.input
    base_config.output_dir = args.output
    
    multi_gpu_config = MultiGPUConfig(
        num_gpus=args.num_gpus,
        max_concurrent_files=args.max_concurrent
    )
    
    # 创建多GPU处理器
    processor = MultiGPULongAudioProcessor(base_config, multi_gpu_config)
    
    # 开始处理
    print("开始多GPU并行处理...")
    results = processor.process_directory_parallel()
    
    # 输出结果
    print(f"\n=== 处理完成 ===")
    print(f"总文件数: {results['total_files']}")
    print(f"成功处理: {results['successful_files']}")
    print(f"失败文件: {results['failed_files']}")
    print(f"成功率: {results['success_rate']:.1f}%")
    print(f"总处理时间: {results['processing_time']/60:.1f}分钟")
    
    print(f"\n=== GPU使用统计 ===")
    for gpu_id, stats in results['gpu_stats'].items():
        print(f"GPU {gpu_id}: 处理了 {stats['processed_count']} 个文件")


if __name__ == "__main__":
    main() 