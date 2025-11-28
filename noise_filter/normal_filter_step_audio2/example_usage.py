#!/usr/bin/env python3
"""
Step-Audio-2 人声检测示例

演示如何使用 Step-Audio-2 判断单个音频文件是否包含人声
参考官方示例: /root/code/github_repos/Step-Audio2/examples.py
"""

import os
import sys
import argparse

# 添加 Step-Audio2 到 Python 路径
step_audio2_dir = '/root/code/github_repos/Step-Audio2'
if os.path.exists(step_audio2_dir):
    sys.path.insert(0, step_audio2_dir)

from stepaudio2 import StepAudio2


def detect_human_voice(audio_file, model_path="/root/code/github_repos/Step-Audio2/Step-Audio-2-mini"):
    """
    使用 Step-Audio-2 判断音频中是否有人声
    
    Args:
        audio_file: 音频文件路径
        model_path: Step-Audio-2 模型路径或模型 ID
    
    Returns:
        bool: True 表示有人声，False 表示无人声
        str: 模型的原始响应
    """
    print(f"\n{'='*60}")
    print(f"检测音频: {audio_file}")
    print(f"{'='*60}")
    
    # 加载模型
    print(f"\n加载 Step-Audio-2 模型...")
    print("(首次加载可能需要下载模型，请耐心等待...)")
    
    model = StepAudio2(model_path)
    
    print("模型加载完成！")
    
    # 准备提示词
    prompt = "Does this audio contain human voice, such as speaking, talking, singing, whispering, laughing, shouting, or any other human vocal sounds? Please answer with ONLY 'yes' or 'no'."
    
    print(f"\n提示词: {prompt}")
    print(f"\n音频文件: {audio_file}")
    print("\n生成回答...")
    
    # 使用 Step-Audio-2 进行音频理解
    # 参考 examples.py 中的 mmau_test
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
    
    # 清理响应
    response_clean = text.strip().lower()
    response_clean = response_clean.replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
    
    # 判断结果
    has_voice = "yes" in response_clean and "no" not in response_clean
    
    # 显示结果
    print(f"\n{'='*60}")
    print(f"原始响应: {text}")
    print(f"清理后: {response_clean}")
    print(f"{'='*60}")
    
    if has_voice:
        print("✓ 结果: 检测到人声")
    else:
        print("✗ 结果: 无人声")
    
    print(f"{'='*60}\n")
    
    return has_voice, text


def main():
    """主函数：测试示例"""
    parser = argparse.ArgumentParser(description="Step-Audio-2 人声检测示例")
    parser.add_argument("audio_file", type=str, help="音频文件路径")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/code/github_repos/Step-Audio2/Step-Audio-2-mini",
        help="模型路径或模型 ID"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.audio_file):
        print(f"错误: 文件不存在: {args.audio_file}")
        return
    
    # 运行检测
    try:
        has_voice, response = detect_human_voice(
            args.audio_file,
            model_path=args.model_path
        )
        
        # 返回状态码
        sys.exit(0 if not has_voice else 1)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()

