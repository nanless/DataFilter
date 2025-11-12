# 修改日志

## 2025-11-12 - Debug 模式添加与 MagicMock 错误修复

### 问题背景

运行 `compute_similarity_prompts.py` 时出现以下错误：
```json
{
  "voiceprint_id": "voiceprint_20250804_1800861",
  "prompt_id": "speechocean762_test_050170001",
  "similarity": -1.0,
  "success": false,
  "error_message": "'<' not supported between instances of 'MagicMock' and 'int'"
}
```

### 根本原因

1. **MagicMock 重复使用**：`compute_similarity_prompts.py` 在模块级和子进程中重复设置 MagicMock，导致某些操作中 Mock 对象被错误使用
2. **环境变量设置时机**：环境变量未在正确的时机设置
3. **缺少 debug 模式**：无法快速测试小批量数据

### 修复内容

#### 1. `multilingual_inference.py` 重大更新

参考新版本（`../tts_speech_voiceprint_filter/multilingual_inference.py`）进行完整重构：

**a) 环境变量设置（模块级别，第 13-20 行）**
```python
os.environ["TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE"] = "1"
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORIO_DISABLE_EXTENSIONS"] = "1"
```

**b) Mock torio 扩展（模块级别，第 34-41 行）**
```python
from unittest.mock import MagicMock

_mock_torio_ext = MagicMock()
_mock_torio_ext.ffmpeg = MagicMock()
sys.modules['torio._extension'] = _mock_torio_ext
sys.modules['torio._extension.ffmpeg'] = _mock_torio_ext.ffmpeg
```

**c) 简化 `_load_model` 方法**
- 移除复杂的异常处理和备用方案
- 直接导入 `wespeaker.cli.speaker.Speaker`
- 在 CPU 模式下设置 `CUDA_VISIBLE_DEVICES`

**d) 更新 `extract_embedding_array` 方法**
- 使用 WeSpeaker 的 `extract_embedding_from_pcm` 接口
- 正确处理 tensor 维度转换
- 确保返回正确的数据类型

#### 2. `compute_similarity.py` 完整重写

参考新版本，添加完整的功能：

**新增功能：**
- `build_pairs_from_mapping`：支持 debug 模式的配对构建
  - 随机打乱 JSON 文件和条目
  - 限制样本数量
  - 详细的进度日志
  
- `main` 函数：完整的命令行接口
  - `--debug`：启用调试模式
  - `--debug_samples`：采样数量（默认 1000）
  - `--debug_dir`：调试输出目录
  
- Debug 模式特性：
  - 随机采样指定数量的配对
  - 保存波形+VAD叠加图
  - 强制单进程避免多GPU开销
  - 默认使用 CPU 提高稳定性

**VAD 支持：**
- `_apply_ten_vad_refined`：refined VAD 掩码生成
- `_save_vad_plot`：保存波形+VAD可视化图
- `_mask_to_segments`：掩码转区间列表

**多进程支持：**
- `_split_even`：均匀分割任务
- `_worker_process`：多GPU工作进程
- `aggregate_and_save`：结果聚合和保存

#### 3. `compute_similarity_prompts.py` 清理

**移除重复代码：**
- 移除模块级别的重复 MagicMock 设置（第 44、54-57 行）
- 更新 `_worker_process_with_env` 函数：
  - 移除子进程中的环境变量重复设置
  - 移除子进程中的 MagicMock 重复设置
  - 简化为只设置 GPU 和导入模型

**保留功能：**
- Debug 模式支持（已存在）
- VAD 参数支持
- 多GPU并行处理

#### 4. `config.json` 更新

保持配置文件格式，确保 debug 配置段完整：
```json
{
  "runtime": {
    "debug": {
      "enabled": false,
      "samples": 100,
      "dir": ""
    }
  }
}
```

#### 5. `run_voiceprint_filter.sh` 验证

验证脚本已包含完整的 debug 模式支持：
- `--debug` 参数
- `--debug_samples` 参数
- `--debug_dir` 参数
- 正确的参数传递和命令构建

### 新增文件

#### 1. `DEBUG_MODE_README.md`
详细的 debug 模式使用文档，包含：
- MagicMock 错误修复说明
- Debug 模式使用方法（Shell 脚本和 Python 直接调用）
- 参数说明
- 输出文件说明
- 常见问题解答
- 性能对比

#### 2. `test_debug_mode.sh`
自动化测试脚本，用于：
- 验证 debug 模式是否正常工作
- 检查 MagicMock 错误是否已修复
- 生成测试报告
- 显示结果摘要

### 测试方法

#### 快速测试（10个样本）
```bash
cd /root/code/github_repos/DataFilter/tts_speech_voiceprint_filter_old
./test_debug_mode.sh
```

#### 手动测试（100个样本）
```bash
./run_voiceprint_filter.sh \
  --prompt_root /root/group-shared/voiceprint/share/voiceclone_child_20250804 \
  --debug \
  --debug_samples 100 \
  --verbose
```

### 预期结果

✅ **成功运行**
- 不再出现 MagicMock 错误
- 成功处理指定数量的样本
- 生成完整的结果 JSON 文件
- 生成 VAD 可视化图（如果启用 debug 模式）
- 相似度计算正常

✅ **输出文件**
- `results/<timestamp>.json`：完整结果
- `results/<timestamp>_filtered_list.txt`：筛除列表
- `<debug_dir>/*.png`：VAD 可视化图（debug 模式）

### 技术要点

1. **环境变量必须在导入 torch 之前设置**
2. **Mock 对象在模块级别设置，所有子进程自动继承**
3. **Debug 模式强制单进程+CPU，避免复杂度**
4. **使用 `extract_embedding_from_pcm` 直接从数组提取 embedding**

### 兼容性

✅ 与新版本（`../tts_speech_voiceprint_filter/`）保持一致
✅ SpeakerIdentify conda 环境
✅ PyTorch 2.7.1
✅ torchaudio 2.7.1
✅ CPU 和 GPU 模式
✅ 单GPU 和多GPU 模式

### 参考文档

- 新版本修复日志：`../tts_speech_voiceprint_filter/FIX_LOG.md`
- Debug 模式说明：`DEBUG_MODE_README.md`
- 测试脚本：`test_debug_mode.sh`
- Shell 脚本：`run_voiceprint_filter.sh`

### 下一步

1. 运行测试脚本验证修复：`./test_debug_mode.sh`
2. 使用 debug 模式检查小批量数据：`./run_voiceprint_filter.sh --debug --debug_samples 100`
3. 确认无错误后，运行完整数据集

### 注意事项

⚠️ **重要**：
- Debug 模式默认使用 CPU，速度较慢但更稳定
- 如需使用 GPU，可显式指定 `--device cuda`
- VAD 图会占用磁盘空间，注意清理
- 首次运行建议使用小样本（10-100）测试

## 修改文件清单

- ✅ `multilingual_inference.py`：完整重构，修复 MagicMock 问题
- ✅ `compute_similarity.py`：完整重写，添加 debug 模式
- ✅ `compute_similarity_prompts.py`：清理重复代码
- ✅ `config.json`：验证配置格式
- ✅ `run_voiceprint_filter.sh`：验证 debug 参数
- ✅ `DEBUG_MODE_README.md`：新增文档
- ✅ `test_debug_mode.sh`：新增测试脚本
- ✅ `CHANGELOG.md`：本文件

## 维护人员

AI Assistant

## 日期

2025-11-12

