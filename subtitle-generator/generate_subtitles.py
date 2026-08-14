#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日语字幕自动生成脚本（方案A：最高准确率）
=============================================
流水线：
  1. 说话人分离 (pyannote-audio)  → 谁在什么时候说话
  2. 语音识别   (faster-whisper)  → 日语语音 → 日语文字
  3. 翻译       (Claude API)      → 日语 → 中文
  4. 生成 SRT   → 原文字幕 + 翻译字幕（按说话人分离）

硬件要求：
  - NVIDIA GPU（8GB+ 显存）推荐，CPU 也可运行（较慢）
  - HuggingFace Token（已内置）用于 pyannote 模型授权

用法：
  python generate_subtitles.py input.mp3
  python generate_subtitles.py input.wav --no-translate        # 只生成日语字幕
  python generate_subtitles.py input.mp3 --speakers 1,2        # 只导出指定说话人
  python generate_subtitles.py input.mp3 --whisper-model medium # 用更小的模型
"""

import os
import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field

# ---- Windows 控制台 UTF-8 ----
if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
# HF Token：从环境变量读取，不硬编码
HF_TOKEN = os.environ.get('HF_TOKEN', '')
if not HF_TOKEN:
    print("[警告] 未设置 HF_TOKEN 环境变量，说话人分离将不可用")
    print("[提示] 设置方法: set HF_TOKEN=hf_你的token")

# 系统提示词：指导 Claude 进行高质量日语→中文字幕翻译
TRANSLATION_SYSTEM_PROMPT = """你是一位专业的日语→中文字幕翻译专家。请将以下日语句子翻译成中文。

翻译要求：
1. **口语化**：字幕是对话/独白，翻译要自然流畅，像真人说话
2. **保留语气**：感叹、犹豫、反问等情绪要保留（如"诶"、"嗯"、"啊"、"呢"、"吧"）
3. **省略补全**：日语常省略主语/宾语，根据上下文合理补充
4. **简洁**：字幕时间有限，翻译要精简但不丢失信息
5. **只输出译文**：不要解释、不要标注，只输出纯中文译文
6. **标点从简**：字幕中尽量不用句号，用空格或换行代替停顿

如果输入是多行句子（每行一句），请逐行翻译，每行一个译文。"""


@dataclass
class SubtitleSegment:
    """单条字幕片段"""
    start: float          # 开始时间（秒）
    end: float            # 结束时间（秒）
    text_ja: str          # 日语原文
    text_zh: str = ''     # 中文翻译
    speaker: str = ''     # 说话人标签
    index: int = 0        # 序号


# ============================================================================
# 步骤 1：说话人分离（Speaker Diarization）
# ============================================================================

def run_diarization(audio_path: str, device: str = 'cuda', num_speakers: int = None) -> list[dict]:
    """
    使用 pyannote-audio 进行说话人分离。

    原理：
      1. VAD（Voice Activity Detection）：找到有语音的片段
      2. Speaker Embedding：将每个小段转为声纹特征向量
      3. Clustering：相似声纹向量归为同一说话人
      4. 输出：每个说话人的时间段

    参数:
        audio_path: 音频文件路径
        device: 'cuda' 或 'cpu'

    返回:
        [{speaker: str, start: float, end: float}, ...]
    """
    print(f"\n{'='*60}")
    print(f"[步骤 1/4] 说话人分离 (Speaker Diarization)")
    print(f"{'='*60}")
    print(f"  模型: pyannote/speaker-diarization-3.1")
    print(f"  设备: {device}")

    from pyannote.audio import Pipeline
    import librosa
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

    if device == 'cuda':
        pipeline = pipeline.to(torch.device('cuda'))

    # ---- torchcodec 不可用时，用 librosa 预加载音频 ----
    print(f"  加载音频...")
    waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, samples)
    audio_input = {'waveform': waveform, 'sample_rate': sample_rate}
    print(f"  波形: {waveform.shape}, 采样率: {sample_rate}")

    print(f"  处理中...（根据音频长度可能需要几分钟）")
    output = pipeline(audio_input, num_speakers=num_speakers if num_speakers else None)

    # pyannote 4.x 返回 DiarizeOutput，通过 .speaker_diarization 获取 Annotation
    diarization_result = output.speaker_diarization

    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end,
        })

    # 按时间排序
    segments.sort(key=lambda x: x['start'])

    # 统计
    speakers = OrderedDict()
    for seg in segments:
        spk = seg['speaker']
        duration = seg['end'] - seg['start']
        speakers[spk] = speakers.get(spk, 0) + duration

    print(f"  检测到 {len(speakers)} 个说话人，共 {len(segments)} 个语音段：")
    for spk, dur in speakers.items():
        print(f"    {spk}: {dur:.1f} 秒 ({dur/60:.1f} 分钟)")

    return segments


# ============================================================================
# 步骤 2：语音识别（ASR with faster-whisper）
# ============================================================================

def run_transcription(
    audio_path: str,
    model_size: str = 'large-v3',
    device: str = 'cuda',
    language: str = 'ja',
    max_chars: int = 40,
) -> list[dict]:
    """
    使用 faster-whisper 进行日语语音识别。

    Whisper 模型大小对比：
      tiny    : ~1GB VRAM,  最快, 准确率 ~85%
      base    : ~1GB VRAM,  较快, 准确率 ~88%
      small   : ~2GB VRAM,  中等, 准确率 ~92%
      medium  : ~5GB VRAM,  较慢, 准确率 ~95%
      large-v3: ~4GB VRAM,  最慢, 准确率 ~97%  ← 推荐

    参数:
        audio_path: 音频文件路径
        model_size: Whisper 模型大小
        device: 'cuda' 或 'cpu'
        language: 语言代码（ja=日语）
        max_chars: 单条字幕最大字符数，超过则按标点拆分

    返回:
        [{start: float, end: float, text: str}, ...]
    """
    print(f"\n{'='*60}")
    print(f"[步骤 2/4] 语音识别 (faster-whisper)")
    print(f"{'='*60}")
    print(f"  模型: whisper {model_size}")
    print(f"  语言: {language}")
    print(f"  设备: {device}")
    print(f"  字幕长度限制: {max_chars} 字符")

    from faster_whisper import WhisperModel

    # 计算类型：GPU 用 float16，CPU 用 int8
    compute_type = 'float16' if device == 'cuda' else 'int8'

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"  转写中...")
    segments_raw, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,     # 静音 300ms 就断句（更短分段）
            min_speech_duration_ms=200,       # 最短有效语音 200ms
            speech_pad_ms=200,                # 语音前后留白
        ),
        word_timestamps=True,                 # 启用词级时间戳
    )

    print(f"  检测到语言: {info.language} (概率: {info.language_probability:.2%})")

    # ---- 收集所有词级时间戳 ----
    all_words = []
    for seg in segments_raw:
        if seg.words:
            for w in seg.words:
                all_words.append({
                    'word': w.word,
                    'start': w.start,
                    'end': w.end,
                    'probability': w.probability,
                })
        else:
            # 没有词级数据时回退：整个 segment 作为一个"词"
            all_words.append({
                'word': seg.text.strip(),
                'start': seg.start,
                'end': seg.end,
                'probability': 1.0,
            })

    print(f"  词级时间戳: {len(all_words)} 个词")
    return all_words  # 返回词列表而非段列表


def split_long_text(text: str, max_chars: int, seg_start: float, seg_end: float) -> list[dict]:
    """
    使用 MeCab 日文分词，在自然词边界处拆分过长字幕。
    避免在助词/助动词处断开（如「し|たくて」→ 保持在一起）。
    """
    import math

    # ---- 尝试用 MeCab 分词（含词性信息） ----
    token_info = []  # [(surface, is_functional), ...]
    try:
        import fugashi
        tagger = fugashi.Tagger()

        # 不应单独出现在行首的功能词
        FUNCTIONAL_POS1 = {'助詞', '助動詞'}

        for w in tagger(text):
            is_func = False
            if w.feature:
                try:
                    is_func = w.feature.pos1 in FUNCTIONAL_POS1
                except Exception:
                    pass
            token_info.append((w.surface, is_func))
    except Exception:
        pass

    if not token_info:
        token_info = [(c, False) for c in text]  # 回退：逐字符

    # ---- 按 max_chars 累积 token，功能词不从行首开始 ----
    parts = []
    current = ''
    for surface, is_func in token_info:
        if len(current) + len(surface) <= max_chars:
            current += surface
        elif is_func and current:
            # 功能词：宁可超一点也不要独立成行
            current += surface
        else:
            if current:
                parts.append(current)
            if len(surface) > max_chars:
                for j in range(0, len(surface), max_chars):
                    parts.append(surface[j:j+max_chars])
                current = ''
            else:
                current = surface
    if current:
        parts.append(current)

    if len(parts) <= 1:
        return [{'start': seg_start, 'end': seg_end, 'text': text}]

    # ---- 均匀分配时间 ----
    duration = seg_end - seg_start
    sub_dur = duration / len(parts)
    results = []
    for i, part in enumerate(parts):
        results.append({
            'start': seg_start + i * sub_dur,
            'end': seg_start + (i + 1) * sub_dur,
            'text': part,
        })
    return results


# ============================================================================
# 步骤 2.5：说话人与转写结果对齐
# ============================================================================

def align_speakers_to_transcript(
    diarization: list[dict],
    words: list[dict],
    max_chars: int = 40,
) -> list[SubtitleSegment]:
    """
    词级对齐：每个词独立匹配说话人，然后按说话人变化重新组段。

    改进点：
      - 每个词按时间中点查找对应说话人（比整段 IoU 精确得多）
      - 检测到说话人变化立即分段 → 解决"一句话两人说话"
      - 同一说话人连续词合并 → 自然组句
    """
    print(f"\n{'='*60}")
    print(f"[步骤 2.5/4] 词级说话人对齐")
    print(f"{'='*60}")

    # ---- 第1步：每个词分配说话人 ----
    for w in words:
        t_mid = (w['start'] + w['end']) / 2
        best_spk = None
        best_overlap = 0.0
        for dia in diarization:
            overlap = min(w['end'], dia['end']) - max(w['start'], dia['start'])
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = dia['speaker']
        if best_spk and best_overlap > 0:
            w['speaker'] = best_spk
        else:
            w['speaker'] = 'Unknown'

    # ---- 第1.5步：中值滤波平滑——孤立词修正为周围词的说话人 ----
    WINDOW = 3
    for i in range(len(words)):
        if words[i]['speaker'] == 'Unknown':
            continue
        # 检查前后窗口内的说话人
        neighbors = []
        for j in range(max(0, i - WINDOW), min(len(words), i + WINDOW + 1)):
            if j != i and words[j]['speaker'] != 'Unknown':
                neighbors.append(words[j]['speaker'])
        if neighbors:
            # 如果当前词是孤立的（与所有邻居不同），改为多数邻居的说话人
            from collections import Counter
            majority = Counter(neighbors).most_common(1)[0][0]
            is_isolated = all(words[i]['speaker'] != n for n in neighbors)
            if is_isolated and len(set(neighbors)) == 1:
                words[i]['speaker'] = majority

    # ---- 第2步：按说话人 + 停顿合并为字幕段 ----
    PAUSE_THRESHOLD = 0.5  # 同一说话人停顿 > 0.5s 则分段
    segments = []
    buf = {'speaker': None, 'tokens': [], 'start': None, 'end': None}

    for w in words:
        gap = w['start'] - buf['end'] if buf['end'] is not None else 0
        speaker_changed = (buf['speaker'] is not None and
                          w['speaker'] != 'Unknown' and
                          buf['speaker'] != 'Unknown' and
                          w['speaker'] != buf['speaker'])

        # 刷新条件：说话人变化 OR (同说话人但长停顿)
        if speaker_changed or (gap > PAUSE_THRESHOLD and buf['tokens']):
            text = ''.join(buf['tokens'])
            if len(text) > max_chars:
                subs = split_long_text(text, max_chars, buf['start'], buf['end'])
                for s in subs:
                    segments.append(SubtitleSegment(
                        start=s['start'], end=s['end'],
                        text_ja=s['text'],
                        speaker=buf['speaker'],
                    ))
            else:
                segments.append(SubtitleSegment(
                    start=buf['start'], end=buf['end'],
                    text_ja=text,
                    speaker=buf['speaker'],
                ))
            buf = {'speaker': None, 'tokens': [], 'start': None, 'end': None}

        if buf['speaker'] is None:
            buf['speaker'] = w['speaker']
            buf['start'] = w['start']
        buf['tokens'].append(w['word'])
        buf['end'] = w['end']

    # 处理最后一段
    if buf['tokens']:
        text = ''.join(buf['tokens'])
        if len(text) > max_chars:
            subs = split_long_text(text, max_chars, buf['start'], buf['end'])
            for s in subs:
                segments.append(SubtitleSegment(
                    start=s['start'], end=s['end'],
                    text_ja=s['text'],
                    speaker=buf['speaker'],
                ))
        else:
            segments.append(SubtitleSegment(
                start=buf['start'], end=buf['end'],
                text_ja=text,
                speaker=buf['speaker'],
            ))

    # ---- 统计 ----
    speaker_counts = OrderedDict()
    for seg in segments:
        speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1

    # ---- 后处理：合并过短片段（< 5 字符）到相邻段 ----
    MIN_CHARS = 5
    merged = []
    for seg in segments:
        if len(seg.text_ja) < MIN_CHARS and merged:
            # 合并到上一段（如果时间间隙小）
            gap = seg.start - merged[-1].end
            if gap < 0.3:
                merged[-1].text_ja += seg.text_ja
                merged[-1].end = seg.end
                continue
        merged.append(seg)

    print(f"  对齐完成（词级）：")
    speaker_counts = OrderedDict()
    for seg in merged:
        speaker_counts[seg.speaker] = speaker_counts.get(seg.speaker, 0) + 1
    for spk, cnt in speaker_counts.items():
        print(f"    {spk}: {cnt} 条")

    return merged


# ============================================================================
# 步骤 3：日语 → 中文翻译（Claude API）
# ============================================================================

def translate_with_claude(
    segments: list[SubtitleSegment],
    api_key: str = None,
    batch_size: int = 20,
) -> list[SubtitleSegment]:
    """
    使用 Claude API 将日语字幕翻译为中文。

    采用批量翻译策略：每 batch_size 条一起发送，减少 API 调用次数。

    参数:
        segments: 待翻译的字幕片段
        api_key: Anthropic API Key（不提供则从环境变量 ANTHROPIC_API_KEY 读取）
        batch_size: 每批翻译的句子数

    返回:
        list[SubtitleSegment]: 翻译后的片段
    """
    print(f"\n{'='*60}")
    print(f"[步骤 3/4] 日语 → 中文翻译 (Claude API)")
    print(f"{'='*60}")

    from anthropic import Anthropic

    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("  [跳过] 未设置 ANTHROPIC_API_KEY，跳过翻译步骤")
        print("  设置方法: set ANTHROPIC_API_KEY=your-key-here")
        return segments

    client = Anthropic(api_key=api_key)

    # ---- 按批次翻译 ----
    total_batches = (len(segments) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(segments))
        batch = segments[start:end]

        # 构建翻译请求：每行一句
        lines = []
        for i, seg in enumerate(batch):
            lines.append(f"[{i+1}] {seg.text_ja}")
        input_text = '\n'.join(lines)

        print(f"  翻译中... 批次 {batch_idx + 1}/{total_batches} ({start+1}-{end}/{len(segments)})")

        try:
            response = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=4096,
                system=TRANSLATION_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': input_text}],
            )

            translated = response.content[0].text.strip()

            # 解析翻译结果：提取译文行
            translated_lines = []
            for line in translated.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 去掉可能的序号前缀 "[1] "
                line = re.sub(r'^\[\d+\]\s*', '', line)
                translated_lines.append(line)

            # 回填翻译结果
            for i, seg in enumerate(batch):
                if i < len(translated_lines):
                    seg.text_zh = translated_lines[i]
                else:
                    seg.text_zh = seg.text_ja  # 翻译缺失时保留原文
                    print(f"    [警告] 第 {start+i+1} 条翻译缺失，保留原文")

        except Exception as e:
            print(f"    [错误] 批次 {batch_idx+1} 翻译失败: {e}")
            for seg in batch:
                seg.text_zh = seg.text_ja  # 翻译失败时保留原文

    translated_count = sum(1 for seg in segments if seg.text_zh and seg.text_zh != seg.text_ja)
    print(f"  翻译完成: {translated_count}/{len(segments)} 条")

    return segments


# ============================================================================
# 步骤 4：生成 SRT 文件
# ============================================================================

def seconds_to_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_by_speaker(
    segments: list[SubtitleSegment],
    output_dir: str,
    prefix: str = 'subtitle',
    lang: str = 'ja',
    target_speakers: set = None,
):
    """
    按说话人分组，为每个说话人生成独立 SRT 文件。

    参数:
        segments: 字幕片段列表
        output_dir: 输出目录
        prefix: 文件名前缀
        lang: 语言标识 ('ja'=日语原文, 'zh'=中文翻译)
        target_speakers: 只导出指定说话人（None=全部导出）
    """
    # ---- 按说话人分组 ----
    speakers = OrderedDict()
    for seg in segments:
        spk = seg.speaker
        if target_speakers and spk not in target_speakers:
            continue
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(seg)

    if not speakers:
        print("  [警告] 没有匹配的说话人数据")
        return

    for spk, spk_segments in speakers.items():
        safe_name = spk.replace(' ', '_').lower()
        filename = f"{prefix}_{safe_name}_{lang}.srt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(spk_segments, start=1):
                text = seg.text_ja if lang == 'ja' else (seg.text_zh or seg.text_ja)
                f.write(f"{i}\n")
                f.write(f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}\n")
                f.write(f"{text}\n")
                f.write("\n")

        print(f"  [OK] {filepath} ({len(spk_segments)} 条)")


def write_combined_srt(
    segments: list[SubtitleSegment],
    output_dir: str,
    prefix: str = 'subtitle',
    lang: str = 'ja',
):
    """
    生成合并字幕（所有说话人混合，按时间排序），标注说话人。

    格式示例：
        说话人1: こんにちは
    """
    # 按开始时间排序
    sorted_segs = sorted(segments, key=lambda x: x.start)

    filename = f"{prefix}_combined_{lang}.srt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(sorted_segs, start=1):
            text = seg.text_ja if lang == 'ja' else (seg.text_zh or seg.text_ja)
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(seg.start)} --> {seconds_to_srt_time(seg.end)}\n")
            f.write(f"[{seg.speaker}] {text}\n")
            f.write("\n")

    print(f"  [OK] {filepath} ({len(sorted_segs)} 条)")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='日语字幕自动生成 —— 说话人分离 + 语音识别 + 翻译 → SRT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python generate_subtitles.py audio.mp3
  python generate_subtitles.py audio.wav --no-translate
  python generate_subtitles.py audio.mp3 --speakers SPEAKER_00,SPEAKER_01
  python generate_subtitles.py audio.mp3 --whisper-model medium --device cpu
  python generate_subtitles.py audio.mp3 --api-key YOUR_API_KEY
        """
    )
    parser.add_argument('audio', help='输入音频文件路径 (.mp3, .wav, .m4a 等)')
    parser.add_argument('-o', '--output-dir', default='./subtitles',
                        help='输出目录（默认: ./subtitles）')
    parser.add_argument('-p', '--prefix', default='subtitle',
                        help='输出文件前缀（默认: subtitle）')
    parser.add_argument('--whisper-model', default='large-v3',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                        help='Whisper 模型大小（默认: large-v3）')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='推理设备（默认: cuda）')
    parser.add_argument('--no-translate', action='store_true',
                        help='跳过翻译步骤，只生成日语字幕')
    parser.add_argument('--api-key', default=None,
                        help='Anthropic API Key（也可通过环境变量 ANTHROPIC_API_KEY 设置）')
    parser.add_argument('--speakers', default=None,
                        help='只导出指定说话人，逗号分隔（如: SPEAKER_00,SPEAKER_01）')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='翻译批量大小（默认: 20）')
    parser.add_argument('--skip-diarization', action='store_true',
                        help='跳过说话人分离（单人音频）')
    parser.add_argument('--num-speakers', type=int, default=None,
                        help='指定说话人数量（如: 2），不指定则自动检测')
    parser.add_argument('--max-chars', type=int, default=40,
                        help='单条字幕最大字符数（默认: 40）')

    args = parser.parse_args()

    # ---- 参数校验 ----
    if not os.path.exists(args.audio):
        print(f"[错误] 文件不存在: {args.audio}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 设备检测 ----
    if args.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                print("[警告] CUDA 不可用，回退到 CPU")
                args.device = 'cpu'
            else:
                print(f"[设备] {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM)")
        except Exception:
            print("[警告] 无法检测 CUDA，使用 CPU")
            args.device = 'cpu'

    print(f"[输入] {args.audio}")
    print(f"[输出] {args.output_dir}/")

    # 解析指定说话人
    target_speakers = None
    if args.speakers:
        target_speakers = set(s.strip() for s in args.speakers.split(','))

    # ========================================================================
    # 步骤 1：说话人分离
    # ========================================================================
    if args.skip_diarization:
        print("\n[步骤 1/4] 说话人分离 → 已跳过（--skip-diarization）")
        diarization = None
    else:
        diarization = run_diarization(args.audio, device=args.device, num_speakers=args.num_speakers)

    # ========================================================================
    # 步骤 2：语音识别
    # ========================================================================
    transcription = run_transcription(
        args.audio,
        model_size=args.whisper_model,
        device=args.device,
        language='ja',
        max_chars=args.max_chars,
    )

    # ========================================================================
    # 步骤 2.5：对齐
    # ========================================================================
    if diarization:
        segments = align_speakers_to_transcript(diarization, transcription, max_chars=args.max_chars)
    else:
        # 无说话人分离，按停顿合并词为段
        words = transcription
        buf = {'text': '', 'start': None, 'end': None}
        segments = []
        for w in words:
            gap = w['start'] - buf['end'] if buf['end'] is not None else 0
            if gap > 0.5 and buf['text']:
                segments.append(SubtitleSegment(
                    start=buf['start'], end=buf['end'],
                    text_ja=buf['text'], speaker='Speaker',
                ))
                buf = {'text': '', 'start': None, 'end': None}
            if buf['start'] is None:
                buf['start'] = w['start']
            buf['text'] += w['word']
            buf['end'] = w['end']
        if buf['text']:
            segments.append(SubtitleSegment(
                start=buf['start'], end=buf['end'],
                text_ja=buf['text'], speaker='Speaker',
            ))
        print("\n[步骤 2.5/4] 说话人分离已跳过，全部归入单个说话人")

    # ========================================================================
    # 步骤 3：翻译
    # ========================================================================
    if args.no_translate:
        print("\n[步骤 3/4] 翻译 → 已跳过（--no-translate）")
    else:
        segments = translate_with_claude(
            segments,
            api_key=args.api_key,
            batch_size=args.batch_size,
        )

    # ========================================================================
    # 步骤 4：生成 SRT 文件
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"[步骤 4/4] 生成 SRT 文件")
    print(f"{'='*60}")

    # 按说话人分离的日语字幕
    print("\n--- 日语原文字幕（按说话人） ---")
    write_srt_by_speaker(
        segments, args.output_dir, args.prefix, lang='ja',
        target_speakers=target_speakers,
    )

    # 按说话人分离的中文字幕
    if not args.no_translate and any(seg.text_zh for seg in segments):
        print("\n--- 中文翻译字幕（按说话人） ---")
        write_srt_by_speaker(
            segments, args.output_dir, args.prefix, lang='zh',
            target_speakers=target_speakers,
        )

    # 合并字幕（含说话人标注）
    print("\n--- 合并字幕 ---")
    write_combined_srt(segments, args.output_dir, args.prefix, lang='ja')
    if not args.no_translate and any(seg.text_zh for seg in segments):
        write_combined_srt(segments, args.output_dir, args.prefix, lang='zh')

    # ========================================================================
    # 汇总
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"[完成] 文件已保存至: {args.output_dir}/")
    print(f"{'='*60}")

    files = os.listdir(args.output_dir)
    for fname in sorted(files):
        fpath = os.path.join(args.output_dir, fname)
        fsize = os.path.getsize(fpath)
        print(f"  [FILE] {fname} ({fsize:,} bytes)")

    return 0


if __name__ == '__main__':
    main()
