#!/usr/bin/env python3
"""
使用 Step-Audio-2 判断音频中是否有人声

参考官方示例: /root/code/github_repos/Step-Audio2/examples.py
使用 StepAudio2 进行推理
"""

import os
import sys
import shutil
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Queue, current_process

# 添加 Step-Audio2 到 Python 路径
step_audio2_dir = '/root/code/github_repos/Step-Audio2'
if os.path.exists(step_audio2_dir):
    sys.path.insert(0, step_audio2_dir)

from stepaudio2 import StepAudio2


def walk_audio_files(folder, queue):
    """遍历文件夹中的所有音频文件"""
    files = []
    for root, dirs, files_in_dir in os.walk(folder):
        for file in files_in_dir:
            if file.endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
                files.append(os.path.join(root, file))
    
    # 将所有文件放入队列
    for file in files:
        queue.put(file)
    queue.put(None)  # 结束信号
    
    return len(files)


def process_files(queue, model_path, device, original_folder, target_folder):
    """处理音频文件的主函数"""
    # 在spawn模式下，需要重新导入sys和设置路径
    import sys
    import os
    
    process_id = current_process().pid
    print(f"进程 {process_id} 启动，使用设备 {device}", flush=True)
    
    # 在spawn模式下，需要重新设置Python路径和导入模块
    step_audio2_dir = '/root/code/github_repos/Step-Audio2'
    if step_audio2_dir not in sys.path:
        sys.path.insert(0, step_audio2_dir)
    
    # 在spawn模式下，需要重新导入torch并设置设备
    import torch
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            print(f"进程 {process_id}: 已设置使用 GPU {device_id}: {torch.cuda.get_device_name(device_id)}", flush=True)
    
    # 在spawn模式下，需要重新导入StepAudio2
    from stepaudio2 import StepAudio2
    
    try:
        # 加载 StepAudio2 模型
        print(f"进程 {process_id}: 加载 Step-Audio-2 模型...", flush=True)
        print(f"进程 {process_id}: 正在加载模型（这可能需要几分钟，请耐心等待）...", flush=True)
        
        model = StepAudio2(model_path)
        
        print(f"进程 {process_id}: 模型加载完成", flush=True)
        
    except Exception as e:
        print(f"进程 {process_id}: 加载模型失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 统计信息
    processed_count = 0
    has_voice_count = 0
    no_voice_count = 0
    error_count = 0
    
    # 使用 position 参数让每个进程的进度条显示在不同行，避免混乱
    # 从 device 中提取 worker_id（用于 position）
    worker_id = 0
    device_label = device
    if device.startswith('cuda:'):
        worker_id = int(device.split(':')[1])
        device_label = f"GPU{worker_id}"
    else:
        device_label = "CPU"
    
    # 不设置 total，让进度条只显示已处理数量
    with tqdm(desc=device_label, unit="文件", position=worker_id, leave=True) as pbar:
        while True:
            audio_file = queue.get()
            if audio_file is None:
                queue.put(None)  # 传递结束信号给下一个进程
                break
            
            try:
                # 使用 Step-Audio-2 进行音频理解
                # 参考 examples.py 中的 uac_test 和 mmau_test
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
                tokens, text, _ = model(messages, max_new_tokens=10, temperature=0.1, do_sample=False)
                
                # 清理响应文本
                response = text.strip().lower()
                response = response.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
                
                # 减少详细输出，只在每100个文件输出一次统计
                if processed_count % 100 == 0 and processed_count > 0:
                    print(f"\n[{device_label}] 已处理: {processed_count}, 有人声: {has_voice_count}, 无人声: {no_voice_count}", flush=True)
                
                # 判断结果并复制文件
                if "no" in response and "yes" not in response:
                    # 没有人声，复制到目标文件夹
                    dest_folder = os.path.dirname(audio_file).replace(original_folder, target_folder)
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_file = os.path.join(dest_folder, os.path.basename(audio_file))
                    shutil.copy(audio_file, dest_file)
                    no_voice_count += 1
                else:
                    has_voice_count += 1
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"\n[{device_label}] 处理文件出错 {audio_file}: {e}", flush=True)
                import traceback
                traceback.print_exc()
            
            finally:
                pbar.update(1)
    
    # 打印统计信息
    print(f"\n[{device_label}] 处理完成:", flush=True)
    print(f"  总处理: {processed_count} 个文件", flush=True)
    print(f"  有人声: {has_voice_count} 个", flush=True)
    print(f"  无人声: {no_voice_count} 个", flush=True)
    print(f"  出错: {error_count} 个", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="使用 Step-Audio-2 检测音频中的人声"
    )
    parser.add_argument(
        "--original_folder", 
        type=str, 
        required=True,
        help="原始音频文件夹路径"
    )
    parser.add_argument(
        "--target_folder", 
        type=str, 
        required=True,
        help="目标文件夹路径（存放无人声的音频）"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/root/code/github_repos/Step-Audio2/Step-Audio-2-mini",
        help="Step-Audio-2 模型路径或模型 ID"
    )
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=1,
        help="并行处理的进程数"
    )
    parser.add_argument(
        "--devices", 
        type=str, 
        default="cuda:0",
        help="使用的设备，多个设备用逗号分隔，例如: cuda:0,cuda:1"
    )
    
    args = parser.parse_args()
    
    # 检查 CUDA 是否可用
    import torch
    if not torch.cuda.is_available():
        print("警告: 未检测到 CUDA 设备，将使用 CPU（速度会很慢）")
        args.devices = "cpu"
    else:
        # 打印 GPU 信息
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"当前使用的 GPU: {device_name}")
    
    # 创建目标文件夹
    os.makedirs(args.target_folder, exist_ok=True)
    
    # 解析设备列表
    devices = args.devices.split(",")
    print(f"将使用以下设备: {devices}")
    
    # 使用 spawn 方法创建多进程上下文（CUDA 需要）
    # 如果使用多进程，必须使用 spawn 方法
    if args.num_processes > 1 and torch.cuda.is_available():
        mp_context = multiprocessing.get_context('spawn')
    else:
        mp_context = multiprocessing.get_context()
    
    # 创建队列
    queue = mp_context.Queue()
    
    # 统计文件总数
    print("统计音频文件...")
    total_files = walk_audio_files(args.original_folder, queue)
    print(f"找到 {total_files} 个音频文件")
    
    if total_files == 0:
        print("未找到音频文件，退出")
        sys.exit(0)
    
    # 根据配置选择处理方式
    if args.num_processes > 1 and torch.cuda.is_available():
        print(f"使用 {args.num_processes} 个进程并行处理")
        
        # 启动多个工作进程（使用 spawn 上下文）
        worker_processes = []
        for i in range(args.num_processes):
            device = devices[i % len(devices)]
            p = mp_context.Process(
                target=process_files,
                args=(
                    queue, 
                    args.model_path,
                    device, 
                    args.original_folder, 
                    args.target_folder
                )
            )
            p.start()
            worker_processes.append(p)
        
        # 等待所有进程完成
        for p in worker_processes:
            p.join()
            
    else:
        # 单进程处理
        print("使用单进程处理")
        process_files(
            queue,
            args.model_path,
            devices[0],
            args.original_folder,
            args.target_folder
        )
    
    print("\n所有文件处理完成！")


if __name__ == "__main__":
    main()

