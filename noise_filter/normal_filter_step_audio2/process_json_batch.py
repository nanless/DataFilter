#!/usr/bin/env python3
"""
使用 Step-Audio-2 批量处理 JSON 文件中的音频列表
读取 merged_noise.json，筛选出没有人声的音频，保存到 merged_noise_nohuman.json

使用 spawn 方式启动多进程，在所有 GPU 上并行处理
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from multiprocessing import Process, Queue, current_process, set_start_method, Manager

# 添加 Step-Audio2 到 Python 路径
step_audio2_dir = '/root/code/github_repos/Step-Audio2'
if os.path.exists(step_audio2_dir):
    sys.path.insert(0, step_audio2_dir)

from stepaudio2 import StepAudio2


def detect_human_voice(audio_file, model):
    """
    使用 Step-Audio-2 判断音频中是否有人声
    
    Args:
        audio_file: 音频文件路径
        model: 已加载的 StepAudio2 模型
    
    Returns:
        bool: True 表示有人声，False 表示无人声
        str: 模型的原始响应
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(audio_file):
            return None, f"文件不存在: {audio_file}"
        
        # 使用 Step-Audio-2 进行音频理解
        prompt = "Does this audio contain human voice, such as speaking, talking, singing, whispering, laughing, shouting, babbling, chuckling, hubbub or any other human vocal sounds, from near or far? You should detect human sound with very high sensitivity and very high recall, and do not let any human vocal sound pass through. Please answer with ONLY 'yes' or 'no'."
        
        messages = [
            {"role": "system", "content": "You are an expert in audio analysis, please analyze the audio content and answer the questions accurately."},
            {"role": "human", "content": [
                {"type": "audio", "audio": audio_file},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": None}
        ]
        
        # 调用模型进行推理
        import sys
        print(f"[detect_human_voice] 开始调用模型推理，文件: {os.path.basename(audio_file)}", file=sys.stderr, flush=True)
        print(f"[detect_human_voice] 消息格式: {messages}", file=sys.stderr, flush=True)
        
        # 使用更短的max_new_tokens，加快推理速度
        # 添加超时保护（通过设置较小的max_new_tokens来加快速度）
        print(f"[detect_human_voice] 调用 model()...", file=sys.stderr, flush=True)
        tokens, text, _ = model(messages, max_new_tokens=10, temperature=0.0, do_sample=False)
        print(f"[detect_human_voice] 模型推理完成，响应: {text[:50] if text else 'None'}", file=sys.stderr, flush=True)
        
        # 清理响应文本
        response = text.strip().lower()
        response = response.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
        
        # 判断结果
        has_voice = "yes" in response and "no" not in response
        
        return has_voice, text
        
    except Exception as e:
        import sys
        import traceback
        print(f"[detect_human_voice] 异常: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return None, f"处理出错: {str(e)}"


def worker_process(worker_id, task_queue, result_queue, model_path, device_id, total_tasks):
    """工作进程：处理音频文件"""
    # 在spawn模式下，需要先导入sys
    import sys
    import os
    
    process_id = current_process().pid
    print(f"[Worker {worker_id}] 进程 {process_id} 启动，使用设备 cuda:{device_id}", flush=True)
    
    # 在spawn模式下，需要重新设置Python路径和导入模块
    step_audio2_dir = '/root/code/github_repos/Step-Audio2'
    if step_audio2_dir not in sys.path:
        sys.path.insert(0, step_audio2_dir)
    
    # 在spawn模式下，需要重新导入torch并设置设备
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        print(f"[Worker {worker_id}] 已设置使用 GPU {device_id}: {torch.cuda.get_device_name(device_id)}", flush=True)
    
    # 在spawn模式下，需要重新导入StepAudio2
    from stepaudio2 import StepAudio2
    
    try:
        # 加载 StepAudio2 模型
        print(f"[Worker {worker_id}] 加载 Step-Audio-2 模型...")
        model = StepAudio2(model_path)
        print(f"[Worker {worker_id}] 模型加载完成")
        
    except Exception as e:
        print(f"[Worker {worker_id}] 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        # 发送错误信号
        result_queue.put(("error", worker_id, str(e)))
        return
    
    # 统计信息
    processed_count = 0
    has_voice_count = 0
    no_voice_count = 0
    error_count = 0
    
    print(f"[Worker {worker_id}] 开始处理任务...", flush=True)
    
    while True:
        # 从队列获取任务
        task = task_queue.get()
        if task is None:  # 结束信号
            print(f"[Worker {worker_id}] 收到结束信号", flush=True)
            break
        
        file_id, file_path = task
        if processed_count == 0:
            print(f"[Worker {worker_id}] 收到第一个任务: {os.path.basename(file_path)}", flush=True)
        if processed_count % 100 == 0:
            print(f"[Worker {worker_id}] 已处理 {processed_count} 个文件，当前: {os.path.basename(file_path)}", flush=True)
        
        try:
            # 检测人声
            print(f"[Worker {worker_id}] 调用 detect_human_voice...", flush=True)
            result, response = detect_human_voice(file_path, model)
            print(f"[Worker {worker_id}] detect_human_voice 返回: result={result}, response={response[:50] if response else None}", flush=True)
            
            if result is None:
                # 处理出错
                error_count += 1
                print(f"[Worker {worker_id}] 处理出错，发送错误结果", flush=True)
                result_queue.put(("error", file_id, file_path, response))
            elif not result:
                # 没有人声
                no_voice_count += 1
                print(f"[Worker {worker_id}] 无人声，发送结果", flush=True)
                result_queue.put(("no_voice", file_id, file_path))
            else:
                # 有人声
                has_voice_count += 1
                print(f"[Worker {worker_id}] 有人声，发送结果", flush=True)
                result_queue.put(("has_voice", file_id, file_path))
            
            processed_count += 1
            print(f"[Worker {worker_id}] 文件处理完成，已处理 {processed_count} 个", flush=True)
            
            # 每处理10个文件打印一次进度（改为10个以便更快看到进度）
            if processed_count % 10 == 0:
                print(f"[Worker {worker_id}] 已处理 {processed_count} 个文件 (有人声: {has_voice_count}, 无人声: {no_voice_count}, 错误: {error_count})", flush=True)
                
        except Exception as e:
            error_count += 1
            error_msg = f"处理文件出错 {file_path}: {e}"
            print(f"\n[Worker {worker_id}] {error_msg}", flush=True)
            import traceback
            traceback.print_exc()
            result_queue.put(("error", file_id, file_path, str(e)))
    
    # 发送统计信息
    result_queue.put(("stats", worker_id, {
        "processed": processed_count,
        "has_voice": has_voice_count,
        "no_voice": no_voice_count,
        "error": error_count
    }))
    
    print(f"\n[Worker {worker_id}] 处理完成:")
    print(f"  总处理: {processed_count} 个文件")
    print(f"  有人声: {has_voice_count} 个")
    print(f"  无人声: {no_voice_count} 个")
    print(f"  出错: {error_count} 个")


def main():
    parser = argparse.ArgumentParser(
        description="使用 Step-Audio-2 批量处理 JSON 文件中的音频列表"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="/root/data/lists/noise/merged_dataset_20251127/merged_noise.json",
        help="输入 JSON 文件路径"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/root/data/lists/noise/merged_dataset_20251127/merged_noise_nohuman.json",
        help="输出 JSON 文件路径（保存没有人声的音频）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/code/github_repos/Step-Audio2/Step-Audio-2-mini",
        help="Step-Audio-2 模型路径"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="使用的 GPU 数量（默认使用所有可用 GPU）"
    )
    
    args = parser.parse_args()
    
    # 检查 CUDA 是否可用（在spawn模式下，需要在主进程导入torch）
    import torch
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA 设备")
        sys.exit(1)
    
    # 获取可用的 GPU 数量
    num_gpus = args.num_gpus if args.num_gpus else torch.cuda.device_count()
    print(f"检测到 {torch.cuda.device_count()} 个 GPU，将使用 {num_gpus} 个 GPU")
    
    # 读取输入 JSON 文件
    print(f"读取输入文件: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    noise_files = data.get('noise_files', [])
    total_files = len(noise_files)
    print(f"找到 {total_files} 个音频文件")
    
    if total_files == 0:
        print("未找到音频文件，退出")
        sys.exit(0)
    
    # 使用 spawn 方式启动多进程（必须在创建Queue之前设置）
    print(f"使用 spawn 方式启动 {num_gpus} 个工作进程...")
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass
    
    # 使用Manager创建队列，确保在spawn模式下能正确工作
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # 将所有任务放入队列
    print("准备任务队列...")
    for item in noise_files:
        file_id = item.get('file_id', '')
        file_path = item.get('path', '')
        if file_path:
            task_queue.put((file_id, file_path))
    
    # 添加结束信号（每个worker一个）
    for _ in range(num_gpus):
        task_queue.put(None)
    
    print(f"任务队列已准备完成，共 {total_files} 个任务")
    
    # 启动工作进程
    worker_processes = []
    for i in range(num_gpus):
        p = Process(
            target=worker_process,
            args=(
                i,
                task_queue,
                result_queue,
                args.model_path,
                i,  # device_id
                total_files  # 用于进度条
            )
        )
        p.start()
        worker_processes.append(p)
    
    # 收集结果
    print("开始处理...")
    no_voice_files = []
    all_stats = {}
    error_files = []
    
    # 创建文件ID到原始数据的映射，加速查找
    file_map = {}
    for item in noise_files:
        file_id = item.get('file_id', '')
        file_path = item.get('path', '')
        if file_id and file_path:
            file_map[(file_id, file_path)] = item
    
    completed_tasks = 0
    last_print_time = 0
    import time
    start_time = time.time()
    
    with tqdm(total=total_files, desc="总进度", unit="文件") as pbar:
        while completed_tasks < total_files:
            try:
                # 使用超时机制，避免永久阻塞
                result = result_queue.get(timeout=300)  # 5分钟超时
                
                if result[0] == "no_voice":
                    _, file_id, file_path = result
                    # 使用映射快速查找
                    key = (file_id, file_path)
                    if key in file_map:
                        no_voice_files.append(file_map[key])
                    completed_tasks += 1
                    pbar.update(1)
                    
                elif result[0] == "has_voice":
                    completed_tasks += 1
                    pbar.update(1)
                    
                elif result[0] == "error":
                    if len(result) == 4:
                        _, file_id, file_path, error_msg = result
                        error_files.append({
                            "file_id": file_id,
                            "path": file_path,
                            "error": error_msg
                        })
                    completed_tasks += 1
                    pbar.update(1)
                    
                elif result[0] == "stats":
                    _, worker_id, stats = result
                    all_stats[worker_id] = stats
                    print(f"\n[Worker {worker_id}] 统计: 处理 {stats['processed']} 个, "
                          f"有人声 {stats['has_voice']} 个, 无人声 {stats['no_voice']} 个, 错误 {stats['error']} 个")
                
                # 每处理1000个文件打印一次进度
                if completed_tasks % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = completed_tasks / elapsed if elapsed > 0 else 0
                    remaining = (total_files - completed_tasks) / rate if rate > 0 else 0
                    print(f"\n进度: {completed_tasks}/{total_files} ({completed_tasks*100/total_files:.1f}%), "
                          f"速度: {rate:.1f} 文件/秒, 预计剩余: {remaining/60:.1f} 分钟")
                    
            except Exception as e:
                # 检查是否是超时异常
                error_str = str(e).lower()
                if "empty" in error_str or "timeout" in error_str:
                    print(f"\n警告: 获取结果超时 (5分钟)，检查工作进程状态...")
                    # 如果超时，检查工作进程是否还在运行
                    alive_count = sum(1 for p in worker_processes if p.is_alive())
                    print(f"存活的工作进程数: {alive_count}/{len(worker_processes)}")
                    if alive_count == 0:
                        print("所有工作进程已结束，但任务未完成，可能存在错误")
                        break
                    else:
                        print("工作进程仍在运行，继续等待...")
                        continue
                else:
                    print(f"\n获取结果时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    # 检查工作进程状态
                    alive_count = sum(1 for p in worker_processes if p.is_alive())
                    print(f"存活的工作进程数: {alive_count}/{len(worker_processes)}")
                    if alive_count == 0:
                        print("所有工作进程已结束，但任务未完成，可能存在错误")
                        break
    
    # 等待所有进程完成
    print("等待所有工作进程完成...")
    for p in worker_processes:
        p.join()
    
    # 计算统计信息
    print("\n计算统计信息...")
    total_duration_seconds = 0.0
    sampling_rate_dist = {}
    
    for item in no_voice_files:
        duration = item.get('duration', 0.0)
        if isinstance(duration, (int, float)):
            total_duration_seconds += float(duration)
        
        sampling_rate = item.get('sampling_rate', None)
        if sampling_rate is not None:
            sr_key = str(sampling_rate)
            sampling_rate_dist[sr_key] = sampling_rate_dist.get(sr_key, 0) + 1
    
    total_duration_hours = total_duration_seconds / 3600.0
    average_duration_seconds = total_duration_seconds / len(no_voice_files) if len(no_voice_files) > 0 else 0.0
    
    # 保存结果
    print(f"\n保存结果到: {args.output_json}")
    output_data = {
        "dataset": data.get('dataset', 'merged_noise') + '_nohuman',
        "total_files": total_files,
        "no_voice_files": len(no_voice_files),
        "has_voice_files": total_files - len(no_voice_files) - len(error_files),
        "error_files": len(error_files),
        "noise_files": no_voice_files,
        "statistics": {
            "total_files": len(no_voice_files),
            "total_duration_seconds": round(total_duration_seconds, 2),
            "total_duration_hours": round(total_duration_hours, 2),
            "average_duration_seconds": round(average_duration_seconds, 3),
            "sampling_rate_distribution": sampling_rate_dist
        }
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"总文件数: {total_files}")
    print(f"无人声文件: {len(no_voice_files)}")
    print(f"有人声文件: {total_files - len(no_voice_files) - len(error_files)}")
    print(f"出错文件: {len(error_files)}")
    print(f"\n统计信息:")
    print(f"  总文件数: {len(no_voice_files)}")
    print(f"  总时长: {round(total_duration_seconds, 2)} 秒 ({round(total_duration_hours, 2)} 小时)")
    print(f"  平均时长: {round(average_duration_seconds, 3)} 秒")
    print(f"  采样率分布: {sampling_rate_dist}")
    print(f"\n结果已保存到: {args.output_json}")
    
    # 打印各worker的统计
    if all_stats:
        print("\n各 Worker 统计:")
        for worker_id, stats in sorted(all_stats.items()):
            print(f"  Worker {worker_id}: 处理 {stats['processed']} 个, "
                  f"有人声 {stats['has_voice']} 个, "
                  f"无人声 {stats['no_voice']} 个, "
                  f"出错 {stats['error']} 个")
    
    if error_files:
        print(f"\n出错文件列表（前10个）:")
        for item in error_files[:10]:
            print(f"  {item['file_id']}: {item['error']}")


if __name__ == "__main__":
    main()

