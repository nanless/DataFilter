# 修复日志：torchaudio 依赖问题

## 问题描述

在 SpeakerIdentify conda 环境下运行声纹筛选脚本时遇到以下错误：

1. **ImportError**: `libtorchaudio.so: cannot open shared object file: No such file or directory`
2. **Segmentation Fault**: 加载 FFmpeg 扩展时在 CPU 模式下崩溃

## 根本原因

- `torchaudio 2.7.1` 安装不完整，缺少 `libtorchaudio.so` 共享库
- `wespeaker` 依赖 `silero_vad`，而 `silero_vad` 导入 `torchaudio`
- `torio` (torchaudio 的 IO 模块) 尝试加载 FFmpeg 扩展，在 CPU 模式下导致段错误

## 解决方案

修改 `multilingual_inference.py`，采用三层防御策略：

### 1. 环境变量设置（模块级别，第 13-20 行）

在导入任何库之前设置环境变量，强制使用 soundfile 后端：

```python
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE"] = "1"
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORIO_DISABLE_EXTENSIONS"] = "1"
```

### 2. Mock torio 扩展（模块级别，第 34-41 行）

预先注入 mock 对象到 `sys.modules`，拦截 torio FFmpeg 扩展的加载：

```python
from unittest.mock import MagicMock

_mock_torio_ext = MagicMock()
_mock_torio_ext.ffmpeg = MagicMock()
sys.modules['torio._extension'] = _mock_torio_ext
sys.modules['torio._extension.ffmpeg'] = _mock_torio_ext.ffmpeg
```

### 3. 简化 _load_model 方法

移除之前复杂的动态 mock 逻辑，只保留必要的 CUDA 设备屏蔽。

## 测试结果

✅ **成功运行**
- 处理完成: 100 对音频对比
- 耗时: 11.81 秒
- 结果文件正确生成
- 无段错误，无导入错误

## 技术细节

### 为什么这个方案有效？

1. **环境变量** 告诉 torchaudio 使用 soundfile 而非原生扩展
2. **Mock 注入** 在 `import wespeaker` 之前就拦截了 torio 的扩展加载
3. **模块级执行** 确保在任何依赖导入之前就完成防御措施

### 为什么不重装 torchaudio？

- 环境中已有其他依赖 torchaudio 的包，重装可能引发兼容性问题
- 当前方案更轻量，不需要修改系统环境
- soundfile 后端功能完全满足声纹识别需求

## 兼容性

- ✅ SpeakerIdentify conda 环境
- ✅ PyTorch 2.7.1
- ✅ torchaudio 2.7.1
- ✅ CPU 模式
- ✅ GPU 模式（未测试但理论兼容）

## 后续建议

如果未来需要使用 torchaudio 的完整功能（如 FFmpeg 读取视频音频），可以考虑：

1. 重新安装 torchaudio: `pip install --force-reinstall torchaudio`
2. 安装系统级 FFmpeg 库: `apt-get install libavutil-dev libavcodec-dev libavformat-dev`
3. 或者使用 conda 安装以确保依赖完整: `conda install -c pytorch torchaudio`

## 修改日期

2025-11-11

## 修改人

AI Assistant

