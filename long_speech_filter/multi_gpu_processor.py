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

from .config import LongAudioProcessingConfig

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
    gpu_memory_fraction: float = 0.7  # 每个GPU使用的显存比例，降低到0.7
    max_concurrent_files: int = 4  # 减少最大并发文件数
    load_balance_strategy: str = "round_robin"  # round_robin, memory_based
    enable_gpu_monitoring: bool = True
    max_processes_per_gpu: int = 1  # 每个GPU最大进程数


class SimpleGPUManager:
    """改进的GPU资源管理器 - 严格限制每GPU一个进程"""
    
    def __init__(self, config: MultiGPUConfig, work_dir: str = "gpu_locks"):
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # 延迟GPU检测，避免在主进程中初始化CUDA
        self.num_gpus = None
        
        print(f"改进的GPU管理器初始化完成，工作目录: {work_dir}")
        print(f"配置: 每GPU最大进程数={config.max_processes_per_gpu}, 显存占用限制={config.gpu_memory_fraction}")
    
    def _detect_gpus(self) -> int:
        """检测可用GPU数量（延迟初始化）"""
        if self.num_gpus is None:
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
                            'process_count': 0,  # 当前进程数
                            'max_processes': self.config.max_processes_per_gpu,
                            'current_processes': [],  # 当前进程列表
                            'processed_count': 0,
                            'last_update': time.time(),
                            'memory_usage': 0.0
                        }))
            except Exception as e:
                print(f"GPU检测失败: {e}")
                self.num_gpus = 0
                
        return self.num_gpus
    
    def acquire_gpu(self, process_id: int, timeout: float = 60.0) -> Optional[int]:
        """获取可用GPU - 严格限制每GPU进程数"""
        num_gpus = self._detect_gpus()
        if num_gpus == 0:
            return None
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 轮询所有GPU寻找可用的
            for gpu_id in range(num_gpus):
                lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
                
                try:
                    with open(lock_file, 'r+') as f:
                        # 尝试获取文件锁
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        # 读取当前状态
                        f.seek(0)
                        data = json.load(f)
                        
                        # 检查是否可以分配更多进程
                        if data['process_count'] < data['max_processes']:
                            # 分配GPU给进程
                            data['process_count'] += 1
                            data['current_processes'].append({
                                'process_id': process_id,
                                'start_time': time.time()
                            })
                            data['last_update'] = time.time()
                            
                            # 写回文件
                            f.seek(0)
                            f.truncate()
                            json.dump(data, f)
                            f.flush()
                            
                            # 释放锁
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            
                            print(f"进程 {process_id} 获取到 GPU {gpu_id} (进程数: {data['process_count']}/{data['max_processes']})")
                            return gpu_id
                        
                        # 释放锁
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        
                except (IOError, OSError, json.JSONDecodeError):
                    # 文件被占用或损坏，跳过
                    continue
            
            time.sleep(0.5)  # 等待时间稍长一点
        
        print(f"进程 {process_id} 在 {timeout}秒内未能获取到GPU")
        return None  # 超时未获取到GPU
    
    def release_gpu(self, gpu_id: int, process_id: int):
        """释放GPU"""
        lock_file = self.work_dir / f"gpu_{gpu_id}.lock"
        
        try:
            with open(lock_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                # 读取当前状态
                f.seek(0)
                data = json.load(f)
                
                # 移除进程
                data['current_processes'] = [
                    p for p in data['current_processes'] 
                    if p['process_id'] != process_id
                ]
                data['process_count'] = len(data['current_processes'])
                data['processed_count'] += 1
                data['last_update'] = time.time()
                
                # 写回文件
                f.seek(0)
                f.truncate()
                json.dump(data, f)
                f.flush()
                
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                print(f"进程 {process_id} 释放 GPU {gpu_id} (剩余进程数: {data['process_count']})")
                
        except (IOError, json.JSONDecodeError) as e:
            print(f"释放GPU {gpu_id} 时出错: {e}")
    
    def update_gpu_memory(self, gpu_id: int, memory_usage: float):
        """更新GPU显存使用情况"""
        lock_file = self.work_dir / f"gpu_{gpu_id}.lock" 
        
        try:
            with open(lock_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                f.seek(0)
                data = json.load(f)
                data['memory_usage'] = memory_usage
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
                    'process_count': 0,
                    'max_processes': self.config.max_processes_per_gpu,
                    'current_processes': [],
                    'processed_count': 0,
                    'last_update': 0,
                    'memory_usage': 0.0
                }
        
        return stats


def cleanup_gpu_memory():
    """清理GPU显存"""
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            # 清理未使用的缓存
            torch.cuda.empty_cache()
            # 强制垃圾回收
            gc.collect()
            # 再次清理缓存
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"清理GPU显存时出错: {e}")


def get_gpu_memory_usage(gpu_id: int) -> float:
    """获取GPU显存使用率"""
    try:
        import torch
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            allocated = torch.cuda.memory_allocated(gpu_id)
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            return allocated / total
    except:
        pass
    return 0.0


def process_single_file_multiprocess(args: Tuple[str, LongAudioProcessingConfig, int, str, float]) -> Dict[str, Any]:
    """多进程处理单个文件的工作函数 - 改进显存管理"""
    file_path, base_config, process_id, work_dir, gpu_memory_fraction = args
    
    import copy
    import time
    import torch
    import gc
    
    # 在子进程中初始化CUDA相关模块
    try:
        # 使用字符串导入避免相对导入问题
        import importlib
        module = importlib.import_module('long_speech_filter.long_audio_processor')
        LongAudioProcessor = getattr(module, 'LongAudioProcessor')
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
    gpu_config = MultiGPUConfig()
    gpu_manager = SimpleGPUManager(gpu_config, work_dir)
    
    logger = logging.getLogger(f"MultiGPU-Process-{process_id}")
    
    start_time = time.time()
    gpu_id = None
    processor = None
    
    try:
        # 获取GPU资源 - 使用进程ID确保唯一性
        gpu_id = gpu_manager.acquire_gpu(process_id)
        if gpu_id is None:
            return {
                'file_path': str(file_path),
                'success': False,
                'gpu_id': None,
                'process_id': process_id,
                'processing_time': time.time() - start_time,
                'error_message': 'GPU资源获取超时 - 所有GPU都被占用'
            }
        
        logger.info(f"进程 {process_id} 获取到 GPU {gpu_id}，开始处理: {Path(file_path).name}")
        
        # 设置CUDA设备和显存管理
        torch.cuda.set_device(gpu_id)
        
        # 设置显存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 清理GPU显存
        cleanup_gpu_memory()
        
        # 监控初始显存使用
        initial_memory = get_gpu_memory_usage(gpu_id)
        logger.info(f"GPU {gpu_id} 初始显存使用率: {initial_memory:.1%}")
        
        # 创建专用于此GPU的配置
        gpu_config = copy.deepcopy(base_config)
        gpu_config.whisper.device = f"cuda:{gpu_id}"
        gpu_config.processing.temp_dir = f"temp_gpu_{gpu_id}_process_{process_id}"
        gpu_config._gpu_device = f"cuda:{gpu_id}"
        gpu_config._process_id = process_id
        
        # 设置更严格的显存限制
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, gpu_id)
        
        # 创建处理器实例
        processor = LongAudioProcessor(gpu_config)
        
        # 监控模型加载后的显存使用
        after_load_memory = get_gpu_memory_usage(gpu_id)
        logger.info(f"GPU {gpu_id} 模型加载后显存使用率: {after_load_memory:.1%}")
        
        # 更新GPU显存状态
        gpu_manager.update_gpu_memory(gpu_id, after_load_memory)
        
        # 处理文件
        result = processor.process_single_audio(str(file_path))
        
        processing_time = time.time() - start_time
        
        # 监控处理完成后的显存使用
        final_memory = get_gpu_memory_usage(gpu_id)
        logger.info(f"GPU {gpu_id} 处理完成后显存使用率: {final_memory:.1%}")
        
        logger.info(f"进程 {process_id} 在 GPU {gpu_id} 上完成处理: {Path(file_path).name} "
                   f"(耗时: {processing_time:.2f}s, 显存: {initial_memory:.1%}→{final_memory:.1%})")
        
        return {
            'file_path': str(file_path),
            'success': result.success,
            'gpu_id': gpu_id,
            'process_id': process_id,
            'processing_time': processing_time,
            'total_segments': result.total_segments,
            'passed_segments': result.passed_segments,
            'output_dirs': result.output_dirs,
            'error_message': result.error_message if not result.success else None,
            'memory_usage': {
                'initial': initial_memory,
                'after_load': after_load_memory,
                'final': final_memory
            }
        }
        
    except Exception as e:
        logger.error(f"进程 {process_id} 在 GPU {gpu_id} 处理文件 {file_path} 失败: {e}")
        logger.exception("详细错误信息:")
        return {
            'file_path': str(file_path),
            'success': False,
            'gpu_id': gpu_id,
            'process_id': process_id,
            'processing_time': time.time() - start_time,
            'error_message': str(e)
        }
    
    finally:
        # 清理资源
        try:
            # 删除处理器实例
            if processor:
                del processor
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理GPU显存
            cleanup_gpu_memory()
            
            # 最终显存检查
            if gpu_id is not None:
                final_cleanup_memory = get_gpu_memory_usage(gpu_id)
                logger.info(f"GPU {gpu_id} 清理后显存使用率: {final_cleanup_memory:.1%}")
                
                # 更新GPU状态
                gpu_manager.update_gpu_memory(gpu_id, final_cleanup_memory)
                
                # 释放GPU资源
                gpu_manager.release_gpu(gpu_id, process_id)
                logger.debug(f"进程 {process_id} 释放 GPU {gpu_id}")
                
        except Exception as cleanup_error:
            logger.error(f"清理资源时出错: {cleanup_error}")


class MultiGPULongAudioProcessor:
    """多GPU长音频处理器 - 改进版本"""
    
    def __init__(self, base_config: LongAudioProcessingConfig, multi_gpu_config: MultiGPUConfig = None):
        self.base_config = base_config
        self.multi_gpu_config = multi_gpu_config or MultiGPUConfig()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建工作目录
        self.work_dir = Path("gpu_work") / f"session_{int(time.time())}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化GPU管理器
        self.gpu_manager = SimpleGPUManager(self.multi_gpu_config, str(self.work_dir))
        
        # 显示GPU信息
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
                self.logger.info(f"配置: 每GPU最大进程数={self.multi_gpu_config.max_processes_per_gpu}")
                
                for i in range(num_gpus):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / 1024**3
                    self.logger.info(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
                    
                # 清理GPU上下文
                cleanup_gpu_memory()
            else:
                self.logger.warning("CUDA不可用")
        except Exception as e:
            self.logger.warning(f"GPU信息获取失败: {e}")
    
    def process_directory_parallel(self) -> Dict[str, Any]:
        """并行处理目录中的所有音频文件 - 改进版本"""
        # 查找音频文件
        audio_files = self._find_audio_files()
        total_files = len(audio_files)
        
        if total_files == 0:
            self.logger.warning(f"在目录 {self.base_config.input_dir} 中未找到音频文件")
            return {'total_files': 0, 'successful_files': 0, 'failed_files': 0, 'results': []}
        
        self.logger.info(f"找到 {total_files} 个音频文件，开始多GPU并行处理")
        
        # 创建输出目录
        Path(self.base_config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 计算合理的进程数 - 严格控制
        estimated_gpus = self.multi_gpu_config.num_gpus
        if estimated_gpus == -1:
            try:
                import torch
                estimated_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except:
                estimated_gpus = 1
        
        # 严格限制并发进程数 = GPU数量 × 每GPU进程数
        max_processes = min(
            estimated_gpus * self.multi_gpu_config.max_processes_per_gpu,
            self.multi_gpu_config.max_concurrent_files,
            total_files
        )
        
        self.logger.info(f"使用 {max_processes} 个并行进程处理 (GPU数量: {estimated_gpus})")
        
        # 准备任务参数
        task_args = [
            (str(file_path), self.base_config, i, str(self.work_dir), self.multi_gpu_config.gpu_memory_fraction)
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
                
                # 分批处理以控制内存使用
                batch_size = max_processes
                for i in range(0, len(task_args), batch_size):
                    batch_args = task_args[i:i + batch_size]
                    
                    self.logger.info(f"处理批次 {i//batch_size + 1}/{(len(task_args) + batch_size - 1)//batch_size} "
                                   f"({len(batch_args)} 个文件)")
                    
                    batch_results = pool.map(process_single_file_multiprocess, batch_args)
                    
                    for result in batch_results:
                        if result['success']:
                            successful_files.append(result)
                        else:
                            failed_files.append(result)
                    
                    # 批次间的内存清理
                    self._cleanup_between_batches()
        
        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"多进程处理过程中发生错误: {e}")
            self.logger.exception("详细错误信息:")
        
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
        
        # 统计显存使用情况
        memory_stats = self._collect_memory_stats(successful_files + failed_files)
        
        # 统计最终结果
        final_stats = {
            'total_files': total_files,
            'successful_files': len(successful_files),
            'failed_files': len(failed_files),
            'success_rate': len(successful_files) / total_files * 100 if total_files > 0 else 0,
            'gpu_stats': self.gpu_manager.get_stats(),
            'memory_stats': memory_stats,
            'processing_time': total_time,
            'average_time_per_file': total_time / total_files if total_files > 0 else 0,
            'successful_results': successful_files,
            'failed_results': failed_files
        }
        
        self.logger.info(f"多GPU处理完成: {final_stats['successful_files']}/{total_files} 文件成功处理 "
                         f"({final_stats['success_rate']:.1f}%) - 总耗时: {total_time/60:.1f}分钟")
        
        # 输出详细统计
        self._log_final_stats(final_stats)
        
        return final_stats
    
    def _cleanup_between_batches(self):
        """批次间清理"""
        try:
            import gc
            gc.collect()
            time.sleep(1)  # 给系统一点时间清理
        except:
            pass
    
    def _collect_memory_stats(self, results: List[Dict]) -> Dict:
        """收集显存使用统计"""
        memory_stats = {
            'peak_usage_by_gpu': {},
            'average_usage_by_gpu': {},
            'memory_efficiency': {}
        }
        
        try:
            gpu_memories = {}
            for result in results:
                if 'memory_usage' in result and result['gpu_id'] is not None:
                    gpu_id = result['gpu_id']
                    if gpu_id not in gpu_memories:
                        gpu_memories[gpu_id] = []
                    gpu_memories[gpu_id].append(result['memory_usage'])
            
            for gpu_id, memories in gpu_memories.items():
                if memories:
                    peak_usage = max(m['final'] for m in memories if 'final' in m)
                    avg_usage = sum(m['final'] for m in memories if 'final' in m) / len(memories)
                    
                    memory_stats['peak_usage_by_gpu'][gpu_id] = peak_usage
                    memory_stats['average_usage_by_gpu'][gpu_id] = avg_usage
                    
        except Exception as e:
            self.logger.warning(f"收集显存统计失败: {e}")
        
        return memory_stats
    
    def _log_final_stats(self, final_stats: Dict):
        """记录最终统计信息"""
        self.logger.info("=== GPU使用统计 ===")
        for gpu_id, stats in final_stats['gpu_stats'].items():
            self.logger.info(f"GPU {gpu_id}: 处理了 {stats['processed_count']} 个文件, "
                           f"最终显存使用: {stats.get('memory_usage', 0):.1%}")
        
        if final_stats['memory_stats']['peak_usage_by_gpu']:
            self.logger.info("=== 显存使用统计 ===")
            for gpu_id, peak in final_stats['memory_stats']['peak_usage_by_gpu'].items():
                avg = final_stats['memory_stats']['average_usage_by_gpu'].get(gpu_id, 0)
                self.logger.info(f"GPU {gpu_id}: 峰值显存 {peak:.1%}, 平均显存 {avg:.1%}")
    
    def _monitor_progress_thread(self, stop_event: threading.Event, total_files: int, start_time: float):
        """改进的进度监控线程"""
        while not stop_event.is_set():
            if stop_event.wait(15):  # 每15秒检查一次
                break
            
            try:
                gpu_stats = self.gpu_manager.get_stats()
                total_processed = sum(stats['processed_count'] for stats in gpu_stats.values())
                
                if total_processed > 0:
                    elapsed_time = time.time() - start_time
                    progress_percent = (total_processed / total_files) * 100
                    estimated_total_time = elapsed_time * total_files / total_processed
                    remaining_time = estimated_total_time - elapsed_time
                    
                    active_gpus = sum(1 for stats in gpu_stats.values() if stats['process_count'] > 0)
                    total_processes = sum(stats['process_count'] for stats in gpu_stats.values())
                    
                    self.logger.info(
                        f"处理进度: {total_processed}/{total_files} ({progress_percent:.1f}%) "
                        f"- 已用时: {elapsed_time/60:.1f}分钟, 预计剩余: {remaining_time/60:.1f}分钟"
                    )
                    self.logger.info(
                        f"GPU状态: {active_gpus}/{len(gpu_stats)} 个GPU活跃, 总进程数: {total_processes}"
                    )
                    
                    # 显示每个GPU的详细状态
                    for gpu_id, stats in gpu_stats.items():
                        if stats['process_count'] > 0:
                            memory_str = f"{stats.get('memory_usage', 0):.1%}" if 'memory_usage' in stats else "N/A"
                            self.logger.info(f"  GPU {gpu_id}: {stats['process_count']} 进程, 显存: {memory_str}")
                            
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
        
        all_files = sorted(audio_files)
        
        # 如果启用跳过已处理文件功能
        if self.base_config.processing.skip_processed and not self.base_config.processing.force_reprocess:
            unprocessed_files = []
            skipped_count = 0
            
            for audio_file in all_files:
                if not self._is_file_processed(audio_file):
                    unprocessed_files.append(audio_file)
                else:
                    skipped_count += 1
            
            if skipped_count > 0:
                self.logger.info(f"跳过 {skipped_count} 个已处理的文件")
                self.logger.info(f"剩余 {len(unprocessed_files)} 个文件需要处理")
            
            return unprocessed_files
        
        return all_files
    
    def _is_file_processed(self, audio_file: Path) -> bool:
        """
        检查音频文件是否已经被处理
        
        判断标准：
        1. 输出目录中存在对应的音频ID目录
        2. 存在 processing_summary.json 文件
        3. summary中显示处理成功
        """
        try:
            audio_id = audio_file.stem
            output_dir = Path(self.base_config.output_dir) / audio_id
            summary_file = output_dir / "processing_summary.json"
            
            # 检查摘要文件是否存在
            if not summary_file.exists():
                return False
            
            # 读取摘要文件检查处理状态
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                
                # 检查是否成功处理且有输出
                if (summary.get('success', False) and 
                    summary.get('processing_results', {}).get('passed_segments', 0) > 0):
                    return True
            
            return False
            
        except Exception as e:
            # 如果检查过程中出现错误，保守起见认为未处理
            self.logger.debug(f"检查文件 {audio_file} 处理状态时出错: {e}")
            return False


def main():
    """主函数，用于测试多GPU处理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="改进的多GPU长音频处理器")
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--num-gpus', type=int, default=-1, help='使用的GPU数量（-1表示全部）')
    parser.add_argument('--max-concurrent', type=int, default=4, help='最大并发文件数')
    parser.add_argument('--processes-per-gpu', type=int, default=1, help='每个GPU的最大进程数')
    parser.add_argument('--memory-fraction', type=float, default=0.7, help='每个GPU使用的显存比例')
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
        max_concurrent_files=args.max_concurrent,
        max_processes_per_gpu=args.processes_per_gpu,
        gpu_memory_fraction=args.memory_fraction
    )
    
    # 创建多GPU处理器
    processor = MultiGPULongAudioProcessor(base_config, multi_gpu_config)
    
    # 开始处理
    print("开始改进的多GPU并行处理...")
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
        memory_str = f", 显存: {stats.get('memory_usage', 0):.1%}" if 'memory_usage' in stats else ""
        print(f"GPU {gpu_id}: 处理了 {stats['processed_count']} 个文件{memory_str}")


if __name__ == "__main__":
    main() 