"""
工具函数和日志配置模块
"""
import os
import logging
import json
import time
from typing import Dict, Any, List
from pathlib import Path
import shutil

def setup_logging(log_file: str = "pipeline.log", log_level: str = "INFO") -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
        
    Returns:
        配置好的logger
    """
    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置根logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化，日志文件：{log_file}")
    
    return logger

def format_duration(seconds: float) -> str:
    """
    将秒数转换为可读的时长字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        可读的时长字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"

def format_file_size(size_bytes: int) -> str:
    """
    将字节数转换为可读的文件大小字符串
    
    Args:
        size_bytes: 字节数
        
    Returns:
        可读的文件大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def validate_input_directory(input_dir: str) -> bool:
    """
    验证输入目录是否有效
    
    Args:
        input_dir: 输入目录路径
        
    Returns:
        是否有效
    """
    if not os.path.exists(input_dir):
        return False
    
    if not os.path.isdir(input_dir):
        return False
    
    return True

def create_output_directory(output_dir: str) -> bool:
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        是否创建成功
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建输出目录失败：{output_dir}，错误：{str(e)}")
        return False

def get_file_size_str(size_bytes: int) -> str:
    """
    将字节数转换为可读的文件大小字符串
    
    Args:
        size_bytes: 字节数
        
    Returns:
        可读的文件大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_duration_str(seconds: float) -> str:
    """
    将秒数转换为可读的时长字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        可读的时长字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 创建目录
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logging.error(f"保存JSON文件失败：{file_path}，错误：{str(e)}")
        return False

def load_json(file_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载JSON文件失败：{file_path}，错误：{str(e)}")
        return {}

def copy_file_with_directory_structure(src_path: str, dest_path: str) -> bool:
    """
    复制文件并保持目录结构
    
    Args:
        src_path: 源文件路径
        dest_path: 目标文件路径
        
    Returns:
        是否复制成功
    """
    try:
        # 创建目标目录
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # 复制文件
        shutil.copy2(src_path, dest_path)
        
        return True
    except Exception as e:
        logging.error(f"复制文件失败：{src_path} -> {dest_path}，错误：{str(e)}")
        return False

def get_directory_size(directory: str) -> int:
    """
    获取目录大小（字节）
    
    Args:
        directory: 目录路径
        
    Returns:
        目录大小（字节）
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    continue
    except Exception:
        pass
    
    return total_size

def count_files_by_extension(directory: str, extensions: List[str]) -> Dict[str, int]:
    """
    按扩展名统计文件数量
    
    Args:
        directory: 目录路径
        extensions: 扩展名列表
        
    Returns:
        扩展名到文件数量的映射
    """
    counts = {ext: 0 for ext in extensions}
    
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in counts:
                    counts[ext] += 1
    except Exception:
        pass
    
    return counts

def clean_temp_files(directory: str, patterns: List[str] = None) -> int:
    """
    清理临时文件
    
    Args:
        directory: 目录路径
        patterns: 要清理的文件模式，默认为常见的临时文件
        
    Returns:
        清理的文件数量
    """
    if patterns is None:
        patterns = ['*.tmp', '*.temp', '*.log', '*.cache', '__pycache__']
    
    cleaned_count = 0
    
    try:
        for pattern in patterns:
            for file_path in Path(directory).glob(f"**/{pattern}"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        cleaned_count += 1
                except Exception:
                    continue
    except Exception:
        pass
    
    return cleaned_count

def format_processing_summary(stats: Dict[str, Any]) -> str:
    """
    格式化处理摘要信息
    
    Args:
        stats: 处理统计信息
        
    Returns:
        格式化的摘要字符串
    """
    summary = []
    summary.append("="*60)
    summary.append("                    处理摘要")
    summary.append("="*60)
    
    # 基本统计
    summary.append(f"总文件数量:          {stats.get('total_files', 0)}")
    summary.append(f"已处理文件:          {stats.get('processed_files', 0)}")
    summary.append(f"通过筛选:            {stats.get('passed_files', 0)}")
    summary.append(f"未通过筛选:          {stats.get('failed_files', 0)}")
    summary.append(f"通过率:              {stats.get('pass_rate', 0):.1f}%")
    
    # 时间统计
    total_time = stats.get('total_processing_time', 0)
    avg_time = stats.get('avg_processing_time', 0)
    summary.append(f"总处理时间:          {get_duration_str(total_time)}")
    summary.append(f"平均处理时间:        {avg_time:.2f}秒/文件")
    
    # 失败原因分析
    summary.append("")
    summary.append("失败原因分析:")
    summary.append(f"  VAD检测失败:       {stats.get('vad_failed', 0)}")
    summary.append(f"  语音识别失败:       {stats.get('transcription_failed', 0)}")
    summary.append(f"  音质评估失败:       {stats.get('quality_failed', 0)}")
    summary.append(f"  语言不匹配:         {stats.get('language_mismatch', 0)}")
    
    summary.append("="*60)
    
    return "\n".join(summary)

def validate_audio_file(file_path: str, supported_formats: List[str]) -> bool:
    """
    验证音频文件是否有效
    
    Args:
        file_path: 音频文件路径
        supported_formats: 支持的格式列表
        
    Returns:
        是否有效
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False
    
    # 检查文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in supported_formats:
        return False
    
    # 检查文件大小
    try:
        size = os.path.getsize(file_path)
        if size == 0:
            return False
    except Exception:
        return False
    
    return True

def create_backup(file_path: str, backup_dir: str = "backup") -> str:
    """
    创建文件备份
    
    Args:
        file_path: 要备份的文件路径
        backup_dir: 备份目录
        
    Returns:
        备份文件路径
    """
    try:
        # 创建备份目录
        os.makedirs(backup_dir, exist_ok=True)
        
        # 生成备份文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # 复制文件
        shutil.copy2(file_path, backup_path)
        
        return backup_path
    except Exception as e:
        logging.error(f"创建备份失败：{file_path}，错误：{str(e)}")
        return ""

def generate_report_html(stats: Dict[str, Any], results: List[Dict[str, Any]], output_path: str) -> bool:
    """
    生成HTML格式的处理报告
    
    Args:
        stats: 统计信息
        results: 处理结果
        output_path: 输出路径
        
    Returns:
        是否生成成功
    """
    try:
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>语音筛选处理报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-item {{ text-align: center; padding: 15px; background: #e8f4f8; border-radius: 5px; }}
        .results {{ margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>语音筛选处理报告</h1>
        <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-item">
            <h3>总文件数</h3>
            <p>{stats.get('total_files', 0)}</p>
        </div>
        <div class="stat-item">
            <h3>通过筛选</h3>
            <p>{stats.get('passed_files', 0)}</p>
        </div>
        <div class="stat-item">
            <h3>未通过筛选</h3>
            <p>{stats.get('failed_files', 0)}</p>
        </div>
        <div class="stat-item">
            <h3>通过率</h3>
            <p>{stats.get('pass_rate', 0):.1f}%</p>
        </div>
    </div>
    
    <div class="results">
        <h2>处理结果详情</h2>
        <table>
            <tr>
                <th>文件路径</th>
                <th>状态</th>
                <th>处理时间</th>
                <th>错误信息</th>
            </tr>
"""
        
        for result in results:
            status_class = "passed" if result.get('passed', False) else "failed"
            status_text = "通过" if result.get('passed', False) else "失败"
            error_msg = result.get('error_message', '') or ''
            
            html_content += f"""
            <tr>
                <td>{result.get('relative_path', '')}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result.get('processing_time', 0):.2f}s</td>
                <td>{error_msg}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
    except Exception as e:
        logging.error(f"生成HTML报告失败：{output_path}，错误：{str(e)}")
        return False 