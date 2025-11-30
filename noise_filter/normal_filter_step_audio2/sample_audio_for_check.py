#!/usr/bin/env python3
"""
从原JSON和nohuman JSON中随机采样音频文件用于检查
- 有人声的音频：在原JSON中但不在nohuman JSON中的
- 无人声的音频：在nohuman JSON中的
- 各随机取100条，复制到目标目录
"""

import os
import sys
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Dict, Set


def load_json(json_path: str) -> Dict:
    """加载JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_set(noise_files: List[Dict]) -> Set[tuple]:
    """从noise_files列表中提取(file_id, path)集合"""
    file_set = set()
    for item in noise_files:
        file_id = item.get('file_id', '')
        file_path = item.get('path', '')
        if file_id and file_path:
            file_set.add((file_id, file_path))
    return file_set


def copy_audio_file(src_path: str, dst_path: str) -> bool:
    """复制音频文件"""
    try:
        # 确保目标目录存在
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
        
        # 复制文件
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"复制文件失败 {src_path} -> {dst_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="从原JSON和nohuman JSON中随机采样音频文件用于检查"
    )
    parser.add_argument(
        "--original_json",
        type=str,
        default="/root/data/lists/noise/merged_dataset_20251127/merged_noise.json",
        help="原始JSON文件路径"
    )
    parser.add_argument(
        "--nohuman_json",
        type=str,
        default="/root/data/lists/noise/merged_dataset_20251127/merged_noise_nohuman.json",
        help="无人声JSON文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录路径"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="每种类型采样数量（默认100）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复性）"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        print(f"使用随机种子: {args.seed}")
    
    # 检查输入文件
    if not os.path.exists(args.original_json):
        print(f"错误: 原始JSON文件不存在: {args.original_json}")
        sys.exit(1)
    
    if not os.path.exists(args.nohuman_json):
        print(f"错误: 无人声JSON文件不存在: {args.nohuman_json}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    has_voice_dir = output_dir / "has_voice"
    no_voice_dir = output_dir / "no_voice"
    has_voice_dir.mkdir(exist_ok=True)
    no_voice_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("加载JSON文件...")
    print("="*60)
    
    # 加载JSON文件
    original_data = load_json(args.original_json)
    nohuman_data = load_json(args.nohuman_json)
    
    original_files = original_data.get('noise_files', [])
    nohuman_files = nohuman_data.get('noise_files', [])
    
    print(f"原始文件总数: {len(original_files)}")
    print(f"无人声文件总数: {len(nohuman_files)}")
    
    # 创建文件映射（用于快速查找）
    original_file_map = {}
    for item in original_files:
        file_id = item.get('file_id', '')
        file_path = item.get('path', '')
        if file_id and file_path:
            original_file_map[(file_id, file_path)] = item
    
    # 获取无人声文件集合
    nohuman_file_set = get_file_set(nohuman_files)
    print(f"无人声文件集合大小: {len(nohuman_file_set)}")
    
    # 找出有人声的文件（在原JSON中但不在nohuman JSON中的）
    has_voice_files = []
    for key, item in original_file_map.items():
        if key not in nohuman_file_set:
            has_voice_files.append(item)
    
    print(f"有人声文件总数: {len(has_voice_files)}")
    print(f"无人声文件总数: {len(nohuman_files)}")
    
    # 随机采样
    print("\n" + "="*60)
    print("随机采样...")
    print("="*60)
    
    # 采样有人声的文件
    if len(has_voice_files) < args.sample_size:
        print(f"警告: 有人声文件数量({len(has_voice_files)})少于采样数量({args.sample_size})，将采样所有文件")
        sampled_has_voice = has_voice_files
    else:
        sampled_has_voice = random.sample(has_voice_files, args.sample_size)
    
    # 采样无人声的文件
    if len(nohuman_files) < args.sample_size:
        print(f"警告: 无人声文件数量({len(nohuman_files)})少于采样数量({args.sample_size})，将采样所有文件")
        sampled_no_voice = nohuman_files
    else:
        sampled_no_voice = random.sample(nohuman_files, args.sample_size)
    
    print(f"采样有人声文件: {len(sampled_has_voice)} 个")
    print(f"采样无人声文件: {len(sampled_no_voice)} 个")
    
    # 复制文件
    print("\n" + "="*60)
    print("复制音频文件...")
    print("="*60)
    
    has_voice_results = []
    no_voice_results = []
    
    # 复制有人声的文件
    print("\n复制有人声文件...")
    for i, item in enumerate(sampled_has_voice, 1):
        src_path = item.get('path', '')
        file_id = item.get('file_id', '')
        
        if not src_path or not os.path.exists(src_path):
            print(f"  跳过 [{i}/{len(sampled_has_voice)}]: 文件不存在 - {src_path}")
            continue
        
        # 获取文件扩展名
        ext = os.path.splitext(src_path)[1] or '.wav'
        dst_filename = f"{file_id}{ext}"
        dst_path = has_voice_dir / dst_filename
        
        if copy_audio_file(src_path, str(dst_path)):
            has_voice_results.append({
                "file_id": file_id,
                "original_path": src_path,
                "copied_path": str(dst_path),
                "duration": item.get('duration', 0.0),
                "sampling_rate": item.get('sampling_rate', None)
            })
            if i % 10 == 0:
                print(f"  已复制 [{i}/{len(sampled_has_voice)}]")
    
    # 复制无人声的文件
    print("\n复制无人声文件...")
    for i, item in enumerate(sampled_no_voice, 1):
        src_path = item.get('path', '')
        file_id = item.get('file_id', '')
        
        if not src_path or not os.path.exists(src_path):
            print(f"  跳过 [{i}/{len(sampled_no_voice)}]: 文件不存在 - {src_path}")
            continue
        
        # 获取文件扩展名
        ext = os.path.splitext(src_path)[1] or '.wav'
        dst_filename = f"{file_id}{ext}"
        dst_path = no_voice_dir / dst_filename
        
        if copy_audio_file(src_path, str(dst_path)):
            no_voice_results.append({
                "file_id": file_id,
                "original_path": src_path,
                "copied_path": str(dst_path),
                "duration": item.get('duration', 0.0),
                "sampling_rate": item.get('sampling_rate', None)
            })
            if i % 10 == 0:
                print(f"  已复制 [{i}/{len(sampled_no_voice)}]")
    
    # 保存结果JSON
    result_json = {
        "original_json": args.original_json,
        "nohuman_json": args.nohuman_json,
        "output_dir": str(output_dir),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "statistics": {
            "total_original_files": len(original_files),
            "total_nohuman_files": len(nohuman_files),
            "total_has_voice_files": len(has_voice_files),
            "sampled_has_voice_count": len(has_voice_results),
            "sampled_no_voice_count": len(no_voice_results)
        },
        "has_voice_samples": has_voice_results,
        "no_voice_samples": no_voice_results
    }
    
    result_json_path = output_dir / "sample_results.json"
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"  有人声文件目录: {has_voice_dir}")
    print(f"  无人声文件目录: {no_voice_dir}")
    print(f"  结果JSON: {result_json_path}")
    print(f"\n统计信息:")
    print(f"  原始文件总数: {len(original_files)}")
    print(f"  无人声文件总数: {len(nohuman_files)}")
    print(f"  有人声文件总数: {len(has_voice_files)}")
    print(f"  成功复制有人声文件: {len(has_voice_results)} 个")
    print(f"  成功复制无人声文件: {len(no_voice_results)} 个")
    
    if len(has_voice_results) < args.sample_size:
        print(f"  警告: 有人声文件复制数量({len(has_voice_results)})少于预期({args.sample_size})")
    if len(no_voice_results) < args.sample_size:
        print(f"  警告: 无人声文件复制数量({len(no_voice_results)})少于预期({args.sample_size})")


if __name__ == "__main__":
    main()

