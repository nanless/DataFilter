# NeMo 文本标准化安装指南

## 安装 NeMo 文本标准化

### 方式1：安装独立包（推荐）

```bash
pip install nemo_text_processing
```

### 方式2：安装完整 NeMo 工具包

```bash
pip install nemo_toolkit[all]
```

## 验证安装

```bash
python3 -c "from nemo_text_processing.text_normalization.normalize import Normalizer; print('NeMo 已安装')"
```

## 使用说明

安装 NeMo 后，SenseVoice 筛选脚本会自动使用 NeMo 的英文文本标准化。

- **英文文本**：使用 NeMo TN（如果已安装）或简单标准化（如果未安装）
- **中文文本**：使用简单标准化（去除标点、空格）

## 回退机制

如果 NeMo 未安装，代码会自动回退到简单标准化方法：
- 英文：转小写、去除标点、标准化空格
- 中文：去除标点符号和空格

这确保了即使没有 NeMo，代码也能正常工作。
