#!/usr/bin/env python3
"""
预训练模型下载脚本
下载Whisper模型到指定目录，以便后续使用
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import whisper
import requests
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_whisper_model(model_name: str, cache_dir: str):
    """
    下载Whisper模型到指定目录
    
    Args:
        model_name: 模型名称
        cache_dir: 缓存目录
    """
    try:
        # whisper模型存放在whisper_modes子目录下
        whisper_cache_dir = os.path.join(cache_dir, 'whisper_modes')
        os.makedirs(whisper_cache_dir, exist_ok=True)
        logger.info(f"创建缓存目录: {whisper_cache_dir}")
        
        # 下载模型
        logger.info(f"开始下载Whisper模型: {model_name}")
        logger.info(f"下载目录: {whisper_cache_dir}")
        
        model = whisper.load_model(model_name, download_root=whisper_cache_dir)
        
        logger.info(f"模型下载成功: {model_name}")
        logger.info(f"模型文件位置: {whisper_cache_dir}")
        
        # 显示模型信息
        logger.info(f"模型类型: {model.dims}")
        logger.info(f"模型语言: {model.is_multilingual}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型下载失败: {str(e)}")
        return False

def download_file_with_progress(url: str, file_path: str, timeout: int = 30):
    """
    下载文件并显示进度
    
    Args:
        url: 下载链接
        file_path: 保存路径
        timeout: 超时时间
    """
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
                    else:
                        print(f"\r  已下载: {downloaded} bytes", end='')
        
        print()  # 换行
        return True
        
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        return False

def download_dnsmos_models(cache_dir: str):
    """
    下载DNSMOS模型到指定目录
    
    Args:
        cache_dir: 缓存目录
    """
    try:
        # 创建DNSMOS模型目录
        dnsmos_dir = Path(cache_dir) / "dnsmos"
        dnsmos_dir.mkdir(parents=True, exist_ok=True)
        
        # DNSMOS模型文件信息
        models = [
            {
                "name": "sig_bak_ovr.onnx",
                "url": "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
                "description": "DNSMOS主模型"
            },
            {
                "name": "model_v8.onnx", 
                "url": "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx",
                "description": "DNSMOS P808模型"
            }
        ]
        
        success_count = 0
        
        for model in models:
            model_path = dnsmos_dir / model["name"]
            
            # 检查模型是否已存在
            if model_path.exists():
                logger.info(f"模型已存在，跳过下载: {model['name']}")
                success_count += 1
                continue
            
            logger.info(f"开始下载 {model['description']}: {model['name']}")
            logger.info(f"下载链接: {model['url']}")
            
            # 下载模型
            if download_file_with_progress(model["url"], str(model_path)):
                logger.info(f"下载成功: {model['name']}")
                success_count += 1
            else:
                logger.error(f"下载失败: {model['name']}")
        
        logger.info(f"DNSMOS模型下载完成: {success_count}/{len(models)}")
        return success_count == len(models)
        
    except Exception as e:
        logger.error(f"DNSMOS模型下载失败: {str(e)}")
        return False

def download_dnsmospro_models(cache_dir: str):
    """
    下载DNSMOSPro模型到指定目录
    
    Args:
        cache_dir: 缓存目录
    """
    try:
        # 创建DNSMOSPro模型目录
        dnsmospro_dir = Path(cache_dir) / "dnsmospro"
        dnsmospro_dir.mkdir(parents=True, exist_ok=True)
        
        # DNSMOSPro模型文件信息
        model_info = {
            "name": "model_best.pt",
            "url": "https://github.com/fcumlin/DNSMOSPro/raw/refs/heads/main/runs/NISQA/model_best.pt",
            "description": "DNSMOSPro NISQA模型"
        }
        
        model_path = dnsmospro_dir / model_info["name"]
        
        # 检查模型是否已存在
        if model_path.exists():
            logger.info(f"模型已存在，跳过下载: {model_info['name']}")
            return True
        
        logger.info(f"开始下载 {model_info['description']}: {model_info['name']}")
        logger.info(f"下载链接: {model_info['url']}")
        
        # 下载模型
        if download_file_with_progress(model_info["url"], str(model_path), timeout=120):
            logger.info(f"下载成功: {model_info['name']}")
            return True
        else:
            logger.error(f"下载失败: {model_info['name']}")
            return False
        
    except Exception as e:
        logger.error(f"DNSMOSPro模型下载失败: {str(e)}")
        return False

def download_whisper_models(models, cache_dir):
    """下载指定的Whisper模型"""
    # whisper模型存放在whisper_modes子目录下
    whisper_cache_dir = os.path.join(cache_dir, 'whisper_modes')
    os.makedirs(whisper_cache_dir, exist_ok=True)
    
    print(f"开始下载Whisper模型到: {whisper_cache_dir}")
    
    for model_name in models:
        try:
            print(f"正在下载Whisper模型: {model_name}")
            
            # 使用whisper.load_model下载模型
            model = whisper.load_model(model_name, download_root=whisper_cache_dir)
            
            # 检查模型是否成功加载
            if model is not None:
                print(f"✅ {model_name} 模型下载成功")
                del model  # 释放内存
            else:
                print(f"❌ {model_name} 模型下载失败")
                
        except Exception as e:
            print(f"❌ 下载 {model_name} 模型时发生错误: {str(e)}")
            continue

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='下载预训练模型到指定目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python download_models.py                          # 下载默认Whisper模型到默认目录
  python download_models.py --model large-v3        # 下载large-v3模型
  python download_models.py --dnsmos                # 仅下载DNSMOS模型
  python download_models.py --all-models            # 下载所有Whisper模型
  python download_models.py --all                   # 下载所有模型（Whisper + DNSMOS）
  python download_models.py --cache-dir /path/to/models  # 指定缓存目录
  python download_models.py --model large-v3 --dnsmos --cache-dir /root/data/pretrained_models
        """
    )
    
    parser.add_argument(
        '--model',
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='要下载的Whisper模型 (默认: large-v3)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='/root/data/pretrained_models',
        help='模型缓存目录 (默认: /root/data/pretrained_models)'
    )
    
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='下载所有可用模型'
    )
    
    parser.add_argument(
        '--dnsmos',
        action='store_true',
        help='下载DNSMOS模型'
    )
    
    parser.add_argument(
        '--dnsmospro',
        action='store_true',
        help='下载DNSMOSPro模型'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='下载所有模型（包括Whisper、DNSMOS和DNSMOSPro）'
    )
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("=" * 60)
    print("              预训练模型下载工具")
    print("=" * 60)
    print(f"缓存目录: {args.cache_dir}")
    
    # 确定要下载的模型
    download_whisper = True
    download_dnsmos = args.dnsmos or args.all
    download_dnsmospro = args.dnsmospro or args.all
    
    if args.all:
        print("下载模式: 所有模型（Whisper + DNSMOS + DNSMOSPro）")
        whisper_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    elif args.all_models:
        print("下载模式: 所有Whisper模型")
        whisper_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    elif args.dnsmos:
        print("下载模式: 仅DNSMOS模型")
        download_whisper = False
        whisper_models = []
    elif args.dnsmospro:
        print("下载模式: 仅DNSMOSPro模型")
        download_whisper = False
        whisper_models = []
        download_dnsmos = False
    else:
        print(f"下载模式: 单个Whisper模型 ({args.model})")
        whisper_models = [args.model]
    
    if download_dnsmos:
        print("包含DNSMOS模型下载")
    if download_dnsmospro:
        print("包含DNSMOSPro模型下载")
    
    print("=" * 60)
    print()
    
    # 检查缓存目录
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.info(f"缓存目录不存在，将创建: {cache_dir}")
    
    # 下载Whisper模型
    whisper_success = 0
    whisper_total = len(whisper_models)
    
    if download_whisper and whisper_models:
        print("正在下载Whisper模型...")
        print("=" * 40)
        
        for model_name in whisper_models:
            print(f"\n正在下载模型: {model_name}")
            print("-" * 40)
            
            if download_whisper_model(model_name, args.cache_dir):
                whisper_success += 1
                print(f"✓ {model_name} 下载成功")
            else:
                print(f"✗ {model_name} 下载失败")
    
    # 下载DNSMOS模型
    dnsmos_success = False
    if download_dnsmos:
        print("\n正在下载DNSMOS模型...")
        print("=" * 40)
        
        dnsmos_success = download_dnsmos_models(args.cache_dir)
        
        if dnsmos_success:
            print("✓ DNSMOS模型下载成功")
        else:
            print("✗ DNSMOS模型下载失败")
    
    # 下载DNSMOSPro模型
    dnsmospro_success = False
    if download_dnsmospro:
        print("\n正在下载DNSMOSPro模型...")
        print("=" * 40)
        
        dnsmospro_success = download_dnsmospro_models(args.cache_dir)
        
        if dnsmospro_success:
            print("✓ DNSMOSPro模型下载成功")
        else:
            print("✗ DNSMOSPro模型下载失败")
    
    # 显示结果
    print("\n" + "=" * 60)
    print("                下载结果")
    print("=" * 60)
    
    if download_whisper and whisper_models:
        print(f"Whisper模型: {whisper_success}/{whisper_total}")
    
    if download_dnsmos:
        print(f"DNSMOS模型: {'成功' if dnsmos_success else '失败'}")
    
    if download_dnsmospro:
        print(f"DNSMOSPro模型: {'成功' if dnsmospro_success else '失败'}")
    
    print(f"缓存目录: {args.cache_dir}")
    
    # 列出已下载的模型
    try:
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            # Whisper模型
            whisper_path = cache_path / "whisper_modes"
            if whisper_path.exists():
                whisper_files = list(whisper_path.glob("*.pt"))
                if whisper_files:
                    print(f"\n已下载的Whisper模型:")
                    for model_file in whisper_files:
                        size = model_file.stat().st_size
                        size_mb = size / (1024 * 1024)
                        print(f"  {model_file.name} ({size_mb:.1f} MB)")
            
            # DNSMOS模型
            dnsmos_path = cache_path / "dnsmos"
            if dnsmos_path.exists():
                dnsmos_files = list(dnsmos_path.glob("*.onnx"))
                if dnsmos_files:
                    print(f"\n已下载的DNSMOS模型:")
                    for model_file in dnsmos_files:
                        size = model_file.stat().st_size
                        size_mb = size / (1024 * 1024)
                        print(f"  {model_file.name} ({size_mb:.1f} MB)")
            
            # DNSMOSPro模型
            dnsmospro_path = cache_path / "dnsmospro"
            if dnsmospro_path.exists():
                dnsmospro_files = list(dnsmospro_path.glob("*.pt"))
                if dnsmospro_files:
                    print(f"\n已下载的DNSMOSPro模型:")
                    for model_file in dnsmospro_files:
                        size = model_file.stat().st_size
                        size_mb = size / (1024 * 1024)
                        print(f"  {model_file.name} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"无法列出模型文件: {str(e)}")
    
    print("=" * 60)
    
    # 检查是否所有模型都下载成功
    all_success = True
    
    if download_whisper and whisper_models:
        if whisper_success != whisper_total:
            all_success = False
    
    if download_dnsmos and not dnsmos_success:
        all_success = False
    
    if download_dnsmospro and not dnsmospro_success:
        all_success = False
    
    if all_success:
        print("\n🎉 所有模型下载完成！")
        print("现在可以使用以下命令运行语音筛选:")
        print(f"python main.py input_dir -o output_dir --model-cache-dir {args.cache_dir}")
        sys.exit(0)
    else:
        print("\n❌ 部分模型下载失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 