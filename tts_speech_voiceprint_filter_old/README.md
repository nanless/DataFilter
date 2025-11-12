### Prompt-vs-Clone 声纹相似度筛选（十个子目录 + 根目录JSON）

本目录提供 Prompt-vs-Clone 相似度计算工具（`compute_similarity_prompts.py`），对十个子目录中的克隆音频与其对应的 prompt 音频计算声纹相似度。

## 一、安装与环境

建议使用 `SpeakerIdentify` conda 环境。需要确保安装：`torch`、`torchaudio`、`tqdm`、`numpy`，并确保本机存在 Multilingual 工程与 `samresnet100` 模型目录。

```bash
conda activate SpeakerIdentify
pip install tqdm torchaudio numpy
# 确保存在：/root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual
```

## 二、使用方法

### Prompt-vs-Clone（十个子目录 + 根目录JSON）

```bash
./run_voiceprint_filter.sh \
  --prompt_root /root/group-shared/voiceprint/share/voiceclone_child_20250804 \
  --threshold 0.90 \
  --model_dir /root/code/gitlab_repos/speakeridentify/InterUttVerify/Multilingual/samresnet100 \
  --num_workers 8 \
  --vad_frame_ms 16 --vad_min_speech_ms 160 --vad_max_silence_ms 200 \
  --verbose
```

## 三、输入与匹配

- Prompt-vs-Clone 目录结构示例：
  - `/root/group-shared/voiceprint/share/voiceclone_child_20250804/`
    - `subdir_1/ ... 音频 ...` ... `subdir_10/ ... 音频 ...`
    - `*.json`（每个 JSON 对应某个子目录，记录 `prompt_id` 与 `clone_id`）
- 匹配策略：
  - 优先在 JSON 同名子目录（如 `abc.json` → `abc/`）中查找 `prompt_id` 与 `clone_id` 音频；
  - 精确 stem 匹配 > 精确 basename 匹配 > 包含关系匹配；
  - 若 `clone_id` 对应音频缺失（常见），跳过该条样本。

## 四、输出

- 结果 JSON：统计信息与每条对比的详细结果
- 筛除列表：`*_filtered_list.txt`，列出不通过的克隆音频路径


