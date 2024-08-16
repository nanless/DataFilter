import os
import sys
from pathlib import Path
import torch
from torchaudio.functional import resample
import soundfile as sf
from multiprocessing import Process, Queue, current_process
from transformers import AutoModelForCausalLM, AutoTokenizer
import string
import shutil
import argparse
from tqdm import tqdm  # 引入tqdm库

AC_REPO_PATH = "/home/kemove/codes/huggingface_repos/efficient_audio_captioning"
sys.path.append(AC_REPO_PATH)

import utils.train_util as train_util

def load_model(cfg, ckpt_path, device):
    model = train_util.init_model_from_config(cfg["model"])
    ckpt = torch.load(ckpt_path, "cpu")
    train_util.load_pretrained_model(model, ckpt)
    model.eval()
    model = model.to(device)
    tokenizer = train_util.init_obj_from_dict(cfg["tokenizer"])
    if not tokenizer.loaded:
        tokenizer.load_state_dict(ckpt["tokenizer"])
    model.set_index(tokenizer.bos, tokenizer.eos, tokenizer.pad)
    return model, tokenizer

def infer(file, runner):
    sr, wav = file
    wav = torch.as_tensor(wav)
    if wav.dtype == torch.short:
        wav = wav / 2 ** 15
    elif wav.dtype == torch.int:
        wav = wav / 2 ** 31
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = resample(wav, sr, runner.target_sr)
    wav_len = len(wav)
    wav = wav.float().unsqueeze(0).to(runner.device)
    input_dict = {
        "mode": "inference",
        "wav": wav,
        "wav_len": [wav_len],
        "specaug": False,
        "sample_method": "beam",
        "beam_size": 3,
    }
    with torch.no_grad():
        output_dict = runner.model(input_dict)
        seq = output_dict["seq"].cpu().numpy()
        cap = runner.tokenizer.decode(seq)[0]
    return cap

class InferRunner:
    def __init__(self, model_name, device):
        self.device = device
        exp_dir = Path(f'{AC_REPO_PATH}/checkpoints') / model_name.lower()
        cfg = train_util.load_config(exp_dir / "config.yaml")
        self.model, self.tokenizer = load_model(cfg, exp_dir / "ckpt.pth", self.device)
        self.target_sr = cfg["target_sr"]

    def change_model(self, model_name):
        exp_dir = Path(f'{AC_REPO_PATH}/checkpoints') / model_name.lower()
        cfg = train_util.load_config(exp_dir / "config.yaml")
        self.model, self.tokenizer = load_model(cfg, exp_dir / "ckpt.pth", self.device)
        self.target_sr = cfg["target_sr"]

def walk_audio_files(folder, queue):
    files = []
    for root, dirs, files_in_dir in os.walk(folder):
        for file in files_in_dir:
            if file.endswith('.wav') or file.endswith('.flac'):
                files.append(os.path.join(root, file))
    
    # 将所有文件放入队列，并添加结束信号
    for file in files:
        queue.put(file)
    queue.put(None)
    
    return len(files)  # 返回文件的总数

def remove_punctuation(input_str):
    return input_str.translate(str.maketrans('', '', string.punctuation))

def process_files(queue, model_name, qwen_model_dir, device_qwen, device_efficient, original_folder, target_folder, total_files):
    infer_runner = InferRunner(model_name, device_efficient)
    process_id = current_process().pid
    print(f"Process {process_id} started")

    # 加载Qwen模型和tokenizer
    model_qwen = AutoModelForCausalLM.from_pretrained(
        qwen_model_dir,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
        cache_dir="downloaded_models"
    ).eval().to(device_qwen)
    tokenizer_qwen = AutoTokenizer.from_pretrained(qwen_model_dir, trust_remote_code=True, cache_dir="downloaded_models")

    with tqdm(total=total_files, desc=f"Process {process_id}") as pbar:  # 使用tqdm创建进度条
        while True:
            audio_file = queue.get()
            if audio_file is None:
                queue.put(None)  # Signal the next process
                break

            try:
                # 生成caption
                wav, sr = sf.read(audio_file)
                caption = infer((sr, wav), infer_runner)
                print(audio_file, "Caption:", caption)
                
                # 使用Qwen LLM判断caption中是否有人声
                prompt = f"According to the caption, is there any people speaking or talking or singing or whispering or laughing in the audio? Caption: {caption}. Answer yes or no only."
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer_qwen.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer_qwen([text], return_tensors="pt").to(device_qwen)

                generated_ids = model_qwen.generate(
                    **model_inputs,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer_qwen.batch_decode(generated_ids, skip_special_tokens=True)[0]
                response = response.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "")

                print("Qwen response:", audio_file, response)

                if "no" in response.lower():
                    dest_folder = os.path.dirname(audio_file).replace(original_folder, target_folder)
                    os.makedirs(dest_folder, exist_ok=True)
                    shutil.copy(audio_file, dest_folder)

            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

            pbar.update(1)  # 更新进度条

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_folder", type=str, default="/data1/data/noise/audioset/selected_filtered_by_qwenaudio_sensevoice")
    parser.add_argument("--target_folder", type=str, default="/data1/data/noise/audioset/selected_filtered_by_qwenaudio_sensevoice_efficientnet_caption")
    parser.add_argument("--model_name", type=str, default="AudioCaps")
    parser.add_argument("--qwen_model_dir", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--devices", type=str, default="cuda:0")
    args = parser.parse_args()

    original_folder = args.original_folder
    target_folder = args.target_folder
    model_name = args.model_name
    qwen_model_dir = args.qwen_model_dir
    num_processes = args.num_processes
    devices = args.devices.split(",")

    torch.manual_seed(1234)

    queue = Queue()

    # Start the file walker process
    walker_process = Process(target=walk_audio_files, args=(original_folder, queue))
    walker_process.start()

    # 获取音频文件总数
    total_files = walk_audio_files(original_folder, queue)

    # 获取当前设备
    current_device = torch.cuda.current_device()
    # 获取设备名称
    device_name = torch.cuda.get_device_name(current_device)
    print(f"当前使用的GPU是: {device_name}")

    # 判断是NVIDIA还是AMD
    if "NVIDIA" in device_name.upper():
        print("这是NVIDIA GPU，使用多线程")
        # Start worker processes
        worker_processes = []
        for i in range(num_processes):
            device_qwen = devices[i % len(devices)]
            device_efficient = devices[i % len(devices)]
            p = Process(target=process_files, args=(queue, model_name, qwen_model_dir, device_qwen, device_efficient, original_folder, target_folder, total_files))
            p.start()
            worker_processes.append(p)

        # Wait for all processes to finish
        walker_process.join()
        for p in worker_processes:
            p.join()
    elif "AMD" in device_name.upper():
        print("这是AMD GPU，使用单线程")
        process_files(queue, model_name, qwen_model_dir, devices[0], devices[0], original_folder, target_folder, total_files)
    else:
        print("未定义的GPU类型")
        exit(1)
