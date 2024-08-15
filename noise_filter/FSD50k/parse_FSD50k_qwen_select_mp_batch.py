import pandas as pd
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import string
import shutil
from multiprocessing import Process, Queue, current_process
import argparse
from tqdm import tqdm

def walk_csv_rows(ground_truth_file_path, queue, batch_size=1):
    batch = []
    with open(ground_truth_file_path, "r") as f:
        reader = list(csv.reader(f))
        total_rows = len(reader)
        for row in tqdm(reader, desc="Reading CSV Rows", total=total_rows):
            batch.append(row)
            if len(batch) == batch_size:
                queue.put(batch)
                batch = []
    if batch:
        queue.put(batch)
    queue.put(None)  # Sentinel to indicate the end of the queue

def process_rows(queue, device, original_folder, model_name, target_folder, batch_size=1):
    process_id = current_process().pid
    print(f"Process {process_id} using device {device}")

    # 清空当前设备的缓存
    torch.cuda.empty_cache()

    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
        cache_dir="downloaded_models"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="downloaded_models", padding_side='left')
    
    while True:
        rows = queue.get()
        if rows is None:
            queue.put(None)  # Signal the next process
            break

        audio_class_dict = {}
        prompts = []
        valid_rows = []

        for row in rows:
            audio_file_path = os.path.join(original_folder, row[0] + ".wav")
            if os.path.exists(audio_file_path):
                real_classs = []
                discriptions = row[1].split(",")
                for description in discriptions:
                    real_classs.append(description.strip())
                audio_class_dict[row[0]] = real_classs
                print(row[0], real_classs)

                prompt = "The classes of one audio clip assigned by human annotators is " + ",".join(real_classs) + ". According to the audio clip classes, is there any human voice in the clip? Answer yes or no only."
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
                valid_rows.append(row)

        if not prompts:
            continue

        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20,
            early_stopping=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for row, response in zip(valid_rows, responses):
            response = response.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            print(row[0], response)

            if response == "no":
                real_classs = audio_class_dict[row[0]]
                selected_class = random.choice(real_classs)
                print("Randomly selected class:", selected_class)

                response = selected_class.lower()

                print("Target folder:", os.path.join(target_folder, response))
                
                if not os.path.exists(os.path.join(target_folder, response)):
                    os.makedirs(os.path.join(target_folder, response))
                
                shutil.copy(audio_file_path, os.path.join(target_folder, response))

        # 清空缓存以防止显存溢出
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_folder", type=str, default="/data1/data/noise/FSD50k/FSD50K.dev_audio")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--ground_truth_file_path", type=str, default="/data1/data/noise/FSD50k/FSD50K.ground_truth/dev.csv")
    parser.add_argument("--target_folder", type=str, default="/data1/data/noise/FSD50k_selected/dev")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--devices", type=str, default="cuda:0")  # 多个设备用逗号分隔,如"cuda:0,cuda:1"
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    original_folder = args.original_folder
    model_name = args.model_name
    ground_truth_file_path = args.ground_truth_file_path
    target_folder = args.target_folder
    num_processes = args.num_processes
    devices = args.devices.split(",")
    batch_size = args.batch_size


    os.makedirs(target_folder, exist_ok=True)
    torch.manual_seed(1234)

    queue = Queue(maxsize=batch_size*num_processes*8)

    # Start the CSV walker process
    walker_process = Process(target=walk_csv_rows, args=(ground_truth_file_path, queue, batch_size))
    walker_process.start()

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"当前使用的GPU是: {device_name}")
        if "NVIDIA" in device_name.upper():
            print("这是NVIDIA GPU，使用多线程")
            worker_processes = []
            for i in range(num_processes):
                p = Process(target=process_rows, args=(queue, devices[i % len(devices)], original_folder, model_name, target_folder, batch_size))
                p.start()
                worker_processes.append(p)
            walker_process.join()
            for p in worker_processes:
                p.join()
        elif "AMD" in device_name.upper():
            print("这是AMD GPU，使用单线程")
            process_rows(queue, devices[0], original_folder, model_name, target_folder, batch_size)
            walker_process.join()
        else:
            print("未定义的GPU类型")
    else:
        print("没有可用的GPU")
        exit(1)
