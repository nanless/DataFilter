import os
from funasr import AutoModel as FunasrAutoModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import shutil
from multiprocessing import Process, Queue, current_process
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def walk_audio_files(folder, queue, batch_size=1):
    batch = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                batch.append(os.path.join(root, file))
                if len(batch) == batch_size:
                    queue.put(batch)
                    batch = []
    if batch:
        queue.put(batch)
    queue.put(None)  # Sentinel to indicate the end of the queue

def process_files(queue, funasr_model_dir, transformers_model_dir, device, original_folder, target_folder, batch_size=1):
    process_id = current_process().pid
    print(f"Process {process_id} using device {device}")

    # 清空当前设备的缓存
    torch.cuda.empty_cache()

    # Load models for this process
    funasr_model = FunasrAutoModel(model=funasr_model_dir, trust_remote_code=True, device=device)
    tokenizer = AutoTokenizer.from_pretrained(transformers_model_dir, trust_remote_code=True, cache_dir="downloaded_models")
    transformers_model = AutoModelForCausalLM.from_pretrained(transformers_model_dir, device_map=None, trust_remote_code=True, cache_dir="downloaded_models").eval().to(device)
    transformers_model.generation_config = GenerationConfig.from_pretrained(transformers_model_dir, trust_remote_code=True)

    total_files = 0
    processed_files = 0

    # 计算文件总数
    for root, dirs, files in os.walk(original_folder):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                total_files += 1

    def create_query(audio_file):
        return tokenizer.from_list_format([
            {'audio': audio_file},
            {'text': 'describe the acoustic environment and acoustic events in the audio in detail in your mind, then tell me, is there any people speaking or talking or singing or whispering or laughing in the audio? just answer only one English word, yes or no, no other words are allowed.'},
        ])

    with tqdm(total=total_files, desc=f"Process {process_id}", unit="file") as pbar, ThreadPoolExecutor(max_workers=batch_size) as executor:
        while True:
            audio_files = queue.get()
            if audio_files is None:
                queue.put(None)  # Signal the next process
                break

            try:
                # queries = list(executor.map(create_query, audio_files))
                queries = [
                    tokenizer.from_list_format([
                        {'audio': audio_file},
                        {'text': 'describe the acoustic environment and acoustic events in the audio in detail in your mind, then tell me, is there any people speaking or talking or singing or whispering or laughing in the audio? just answer only one English word, yes or no, no other words are allowed.'},
                    ])
                    for audio_file in audio_files
                ]
                qwen_responses = [transformers_model.chat(tokenizer, query=query, history=None)[0] for query in queries]
                # Second check with funasr model
                funasr_res = funasr_model.generate(
                    input=audio_files,
                    cache={},
                    language="auto",
                    use_itn=False,
                    batch_size=batch_size,
                    disable_pbar=True
                )

                for audio_file, qwen_response, funasr_res in zip(audio_files, qwen_responses, funasr_res):
                    print(audio_file, qwen_response, funasr_res['text'])
                    qwen_response = qwen_response.strip().lower()
                    if "no" in qwen_response:
                        if "|nospeech|" in funasr_res['text']:
                            dest_folder = os.path.dirname(audio_file).replace(original_folder, target_folder)
                            os.makedirs(dest_folder, exist_ok=True)
                            shutil.copy(audio_file, dest_folder)
                    processed_files += 1
                    pbar.update(1)
            except Exception as e:
                print(e)

            # 清空缓存以防止显存溢出
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_folder", type=str, default="/data1/data/noise/audioset/selected")
    parser.add_argument("--target_folder", type=str, default="/data1/data/noise/audioset/selected_filtered_by_qwenaudio_sensevoice")
    parser.add_argument("--funasr_model_dir", type=str, default="iic/SenseVoiceSmall")
    parser.add_argument("--transformers_model_dir", type=str, default="Qwen/Qwen-Audio-Chat")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--devices", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    original_folder = args.original_folder
    target_folder = args.target_folder
    funasr_model_dir = args.funasr_model_dir
    transformers_model_dir = args.transformers_model_dir
    num_processes = args.num_processes
    devices = args.devices.split(",")
    batch_size = args.batch_size

    os.makedirs(target_folder, exist_ok=True)
    torch.manual_seed(1234)

    queue = Queue(maxsize=batch_size*num_processes*8)

    # Start the file walker process
    walker_process = Process(target=walk_audio_files, args=(original_folder, queue, batch_size))
    walker_process.start()

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"当前使用的GPU是: {device_name}")
        if "NVIDIA" in device_name.upper():
            print("这是NVIDIA GPU，使用多线程")
            worker_processes = []
            for i in range(num_processes):
                p = Process(target=process_files, args=(queue, funasr_model_dir, transformers_model_dir, devices[i % len(devices)], original_folder, target_folder, batch_size))
                p.start()
                worker_processes.append(p)
            walker_process.join()
            for p in worker_processes:
                p.join()
        elif "AMD" in device_name.upper():
            print("这是AMD GPU，使用单线程")
            process_files(queue, funasr_model_dir, transformers_model_dir, devices[0], original_folder, target_folder, batch_size)
        else:
            print("未定义的GPU类型")
    else:
        print("没有可用的GPU")
        exit(1)
