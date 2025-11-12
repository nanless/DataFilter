# 快速开始

## 🎯 快速测试（推荐）

修复已完成！运行以下命令立即测试：

```bash
cd /root/code/github_repos/DataFilter/tts_speech_voiceprint_filter_old
./test_debug_mode.sh
```

这将：
- ✅ 验证 MagicMock 错误已修复
- ✅ 使用 10 个样本快速测试
- ✅ 生成 VAD 可视化图
- ✅ 显示结果摘要

## 🚀 使用 Debug 模式（100个样本）

```bash
./run_voiceprint_filter.sh \
  --prompt_root /root/group-shared/voiceprint/share/voiceclone_child_20250804 \
  --debug \
  --debug_samples 100 \
  --verbose
```

## 📝 主要修复

1. **MagicMock 错误** ✅ 已修复
   - 更新 `multilingual_inference.py`
   - 移除重复的 Mock 代码
   
2. **Debug 模式** ✅ 已添加
   - 支持小批量测试（默认100个样本）
   - 自动生成 VAD 可视化图
   - 详细的调试信息

3. **环境兼容性** ✅ 已优化
   - CPU 模式更稳定
   - GPU 模式性能更好
   - 多GPU 并行支持

## 📖 详细文档

- **修复详情**：`CHANGELOG.md`
- **使用说明**：`DEBUG_MODE_README.md`
- **测试脚本**：`test_debug_mode.sh`

## ⚡ 性能参考

| 模式 | 样本数 | 设备 | 预计时间 |
|------|--------|------|----------|
| 测试 | 10 | CPU | ~10秒 |
| Debug | 100 | CPU | ~1-2分钟 |
| Debug | 100 | GPU | ~30秒 |
| 生产 | 全部 | 8xGPU | ~5-30分钟 |

## ❓ 常见问题

### Q: 测试失败怎么办？
A: 检查根目录路径是否正确，修改 `test_debug_mode.sh` 中的 `PROMPT_ROOT` 变量

### Q: 需要安装额外的依赖吗？
A: 不需要，使用现有的 SpeakerIdentify conda 环境即可

### Q: debug 模式可以使用 GPU 吗？
A: 可以，添加 `--device cuda` 参数

## 🎉 现在就开始

```bash
# 1. 进入目录
cd /root/code/github_repos/DataFilter/tts_speech_voiceprint_filter_old

# 2. 运行测试
./test_debug_mode.sh

# 3. 查看结果
ls -lh test_output/
```

## 📞 需要帮助？

查看详细文档：
- `DEBUG_MODE_README.md`：完整的使用指南
- `CHANGELOG.md`：修改详情和技术说明

