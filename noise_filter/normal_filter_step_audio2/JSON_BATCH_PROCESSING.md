# JSON 批量处理说明

## 功能

读取 `merged_noise.json` 文件，使用 Step-Audio-2 模型检测所有音频文件中的人声，将没有人声的音频保存到 `merged_noise_nohuman.json`。

## 使用方法

### 方法 1: 使用运行脚本（推荐）

```bash
cd /root/code/github_repos/DataFilter/noise_filter/normal_filter_step_audio2
./run_json_batch.sh
```

### 方法 2: 直接使用 Python

```bash
# 激活环境
conda activate stepaudio2
export PYTHONPATH="/root/code/github_repos/Step-Audio2:${PYTHONPATH}"

# 运行脚本
cd /root/code/github_repos/DataFilter/noise_filter/normal_filter_step_audio2
python process_json_batch.py \
    --input_json /root/data/lists/noise/merged_dataset_20251127/merged_noise.json \
    --output_json /root/data/lists/noise/merged_dataset_20251127/merged_noise_nohuman.json \
    --model_path /root/code/github_repos/Step-Audio2/Step-Audio-2-mini
```

## 参数说明

- `--input_json`: 输入 JSON 文件路径（默认: `/root/data/lists/noise/merged_dataset_20251127/merged_noise.json`）
- `--output_json`: 输出 JSON 文件路径（默认: `/root/data/lists/noise/merged_dataset_20251127/merged_noise_nohuman.json`）
- `--model_path`: Step-Audio-2 模型路径（默认: `/root/code/github_repos/Step-Audio2/Step-Audio-2-mini`）
- `--num_gpus`: 使用的 GPU 数量（默认: 使用所有可用 GPU）

## 输入 JSON 格式

```json
{
  "dataset": "merged_noise",
  "noise_files": [
    {
      "file_id": "DEMAND-noise_segments_10s_DKITCHEN_ch01_seg000",
      "path": "/path/to/audio.wav",
      "duration": 10.0,
      "sampling_rate": 16000
    },
    ...
  ]
}
```

## 输出 JSON 格式

```json
{
  "dataset": "merged_noise_nohuman",
  "total_files": 1000,
  "no_voice_files": 650,
  "has_voice_files": 340,
  "error_files": 10,
  "noise_files": [
    {
      "file_id": "DEMAND-noise_segments_10s_DKITCHEN_ch01_seg000",
      "path": "/path/to/audio.wav",
      "duration": 10.0,
      "sampling_rate": 16000
    },
    ...
  ]
}
```

## 特性

- ✅ 使用 **spawn** 方式启动多进程（适合 CUDA）
- ✅ 自动检测并使用所有可用 GPU
- ✅ 每个 GPU 独立加载模型，并行处理
- ✅ 实时显示处理进度
- ✅ 详细的统计信息输出
- ✅ 错误处理：记录处理失败的文件

## 处理流程

1. **读取输入 JSON**: 加载所有音频文件列表
2. **分配任务**: 将任务分配到所有 GPU
3. **并行处理**: 每个 GPU 独立处理分配的任务
4. **收集结果**: 汇总所有处理结果
5. **保存输出**: 将没有人声的音频保存到输出 JSON

## 性能

- 每个 GPU 独立加载模型，互不干扰
- 使用队列机制，自动负载均衡
- 支持处理大量文件（百万级）

## 注意事项

1. **显存要求**: 每个 GPU 需要足够的显存加载模型（建议 >= 8GB）
2. **处理时间**: 取决于文件数量和 GPU 数量
3. **错误处理**: 处理失败的文件会记录在统计信息中，但不会包含在输出 JSON 中
4. **spawn 模式**: 使用 spawn 方式启动进程，确保 CUDA 正常工作

## 故障排除

### 显存不足

如果遇到显存不足错误：
- 减少使用的 GPU 数量：`--num_gpus 4`
- 确保没有其他程序占用 GPU

### 进程启动失败

如果进程启动失败：
- 检查 stepaudio2 环境是否正确激活
- 检查 PYTHONPATH 是否正确设置
- 检查模型路径是否存在

### 处理速度慢

- 确保使用 GPU 而不是 CPU
- 检查 GPU 使用率：`nvidia-smi`
- 确保没有其他程序占用 GPU

