#!/bin/bash

# 测试增量处理功能的示例脚本

echo "=================================="
echo "TTS筛选增量处理功能说明"
echo "=================================="
echo ""

echo "🎯 核心功能："
echo "  - 读取已有的JSON结果文件"
echo "  - 自动识别已处理的音频文件"
echo "  - 只处理新增的音频文件"
echo "  - 将新结果合并到原有JSON文件中"
echo "  - 更新统计信息"
echo ""

echo "📌 使用方法示例："
echo ""

echo "1. 默认行为（增量处理模式）："
echo "   ./start_filter_all.sh"
echo "   说明：如果结果文件已存在，只处理新增的音频"
echo ""

echo "2. 强制重新处理所有音频："
echo "   ./start_filter_all.sh --force"
echo "   或"
echo "   ./start_filter_all.sh --no-skip_existing"
echo "   说明：忽略已有结果，重新处理所有音频"
echo ""

echo "3. 处理特定模式的文件（增量）："
echo "   ./start_filter_all.sh --pattern 'voiceprint_20250804_part*_*.json'"
echo ""

echo "4. 组合使用："
echo "   ./start_filter_all.sh --pattern 'voiceprint_20250804_part5_*.json' --force --num_gpus 4"
echo ""

echo "5. 单个文件增量处理："
echo "   ./run_single_tts_filter.sh /path/to/base_dir /path/to/json_file.json --output results.json"
echo ""

echo "6. 单个文件强制重新处理："
echo "   ./run_single_tts_filter.sh /path/to/base_dir /path/to/json_file.json --output results.json --force"
echo ""

echo "=================================="
echo "💡 功能特点："
echo "=================================="
echo "✅ 增量处理："
echo "   - 自动检测已处理的音频"
echo "   - 跳过已处理的音频，节省时间"
echo "   - 新旧结果自动合并"
echo ""
echo "✅ 统计更新："
echo "   - 累计显示所有处理结果"
echo "   - 显示跳过的音频数量"
echo "   - CER统计包含所有数据"
echo ""
echo "✅ 灵活控制："
echo "   - 支持增量处理（默认）"
echo "   - 支持强制重新处理"
echo "   - 保持数据完整性"
echo ""

echo "查看帮助信息："
echo "  ./run_all_tts_filter.sh --help"
echo "  ./run_single_tts_filter.sh --help"
