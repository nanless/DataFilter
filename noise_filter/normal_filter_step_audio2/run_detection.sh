#!/bin/bash

# Step-Audio-2 人声检测脚本运行示例
# 使用 Step-Audio-2 判断音频中是否有人声

# ============================================
# 配置参数
# ============================================

# 原始音频文件夹（必需修改）
ORIGINAL_FOLDER="/root/group-shared/voiceprint/data/multimodal/audio-visual/AudioSet/agkphysics___audio_set/balanced/0.0.0/wavs"

# 目标文件夹（存放无人声的音频）（必需修改）
TARGET_FOLDER="/root/group-shared/voiceprint/data/multimodal/audio-visual/AudioSet/agkphysics___audio_set/balanced/0.0.0/wavs_nohuman"

# 模型配置（使用本地已下载的模型）
MODEL_PATH="/root/code/github_repos/Step-Audio2/Step-Audio-2-mini"  # 本地模型路径

# 并行处理配置
NUM_PROCESSES=4  # 进程数（根据 GPU 数量调整）
DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"  # 使用的 GPU 设备，多个设备用逗号分隔，如 "cuda:0,cuda:1"

# ============================================
# 检查参数
# ============================================

# 检查原始文件夹是否存在
if [ ! -d "$ORIGINAL_FOLDER" ]; then
    echo "错误: 原始音频文件夹不存在: $ORIGINAL_FOLDER"
    exit 1
fi

# 创建目标文件夹
mkdir -p "$TARGET_FOLDER"

# ============================================
# 打印配置信息
# ============================================

echo "======================================"
echo "Step-Audio-2 人声检测脚本"
echo "======================================"
echo "原始文件夹: $ORIGINAL_FOLDER"
echo "目标文件夹: $TARGET_FOLDER"
echo "模型路径: $MODEL_PATH"
echo "并行进程数: $NUM_PROCESSES"
echo "使用设备: $DEVICES"
echo "======================================"
echo ""

# ============================================
# 运行脚本
# ============================================

python step_audio2_human_voice_detector.py \
    --original_folder "$ORIGINAL_FOLDER" \
    --target_folder "$TARGET_FOLDER" \
    --model_path "$MODEL_PATH" \
    --num_processes "$NUM_PROCESSES" \
    --devices "$DEVICES"

# ============================================
# 显示结果
# ============================================

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "处理完成！"
    echo "======================================"
    echo "无人声音频已保存到: $TARGET_FOLDER"
else
    echo ""
    echo "======================================"
    echo "处理失败，请检查错误信息"
    echo "======================================"
    exit 1
fi

