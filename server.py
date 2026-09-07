#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字幕生成器 - REST API 后端"""
import os, sys, re, json, tempfile, uuid, threading, time
from pathlib import Path
from collections import OrderedDict

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# 加载 .env 配置（Token、API Key 等）
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_subtitles import (
    run_diarization, run_transcription, align_speakers_to_transcript,
    seconds_to_srt_time,
)

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# 增加上传大小限制
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 主线程预加载 torch，避免子线程卡死
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    DEVICE = 'cpu'

# 任务状态存储
tasks = {}
processing_lock = threading.Lock()  # 保证同一时间只处理一个任务（GPU 显存有限）

@app.errorhandler(Exception)
def handle_error(e):
    """全局错误处理，确保始终返回 JSON"""
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500


# ============================================================================
# 翻译引擎
# ============================================================================

SYSTEM_PROMPT = '专业日语→中文字幕翻译。口语化、自然流畅。每行只输出中文译文。'


def _translate_claude(entries, api_key, tasks, task_id):
    """Claude API 翻译，支持自定义 Key"""
    from anthropic import Anthropic
    # 优先用前端传入的 key，其次环境变量，最后自动认证
    key = api_key or os.environ.get('ANTHROPIC_API_KEY', '')
    client = Anthropic(api_key=key) if key else Anthropic()
    BATCH = 30
    for bi in range(0, len(entries), BATCH):
        batch = entries[bi:bi + BATCH]
        pct = 75 + 20 * (bi + 1) / len(entries)
        tasks[task_id]['progress'] = int(pct)
        tasks[task_id]['message'] = f'Claude 翻译中... {bi+1}-{min(bi+BATCH, len(entries))}/{len(entries)}'
        lines = '\n'.join(f'[{i+1}] {e["text_ja"]}' for i, e in enumerate(batch))
        try:
            resp = client.messages.create(
                model='claude-sonnet-4-6', max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': lines}],
            )
            tb = [b for b in resp.content if b.type == 'text']
            trans = [re.sub(r'^\[\d+\]\s*', '', t).strip()
                     for t in tb[0].text.strip().split('\n') if t.strip()] if tb else []
        except Exception as ex:
            trans = []
            # 首次失败时记录原因
            if bi == 0:
                tasks[task_id]['message'] = f'翻译失败: {str(ex)[:80]}'
        for i, e in enumerate(batch):
            e['text_zh'] = trans[i] if i < len(trans) else e['text_ja']


def _translate_google(entries, tasks, task_id):
    """Google Translate 免费翻译（无需 Key，国内可能不可用）"""
    from deep_translator import GoogleTranslator
    fail_count = 0
    for i, e in enumerate(entries):
        if i % 5 == 0:
            pct = 75 + 20 * (i + 1) / len(entries)
            tasks[task_id]['progress'] = int(pct)
            tasks[task_id]['message'] = f'Google 翻译中... {i+1}/{len(entries)}'
        try:
            e['text_zh'] = GoogleTranslator(source='ja', target='zh-CN').translate(e['text_ja'])
        except Exception:
            e['text_zh'] = e['text_ja']
            fail_count += 1
            if fail_count >= 3:
                # 连续失败 3 次说明 Google 不可用，后续全跳过
                for j in range(i + 1, len(entries)):
                    entries[j]['text_zh'] = entries[j]['text_ja']
                tasks[task_id]['message'] = 'Google 不可用（已被墙），已跳过翻译'
                return


def _translate_baidu(entries, tasks, task_id):
    """百度翻译 API（国内可用，免费额度）"""
    import hashlib
    import requests as _req

    appid = os.environ.get('BAIDU_APPID', '')
    key = os.environ.get('BAIDU_API_KEY', '')
    if not appid or not key:
        for e in entries:
            e['text_zh'] = e['text_ja']
        tasks[task_id]['message'] = '百度翻译未配置（需 BAIDU_APPID 和 BAIDU_API_KEY）'
        return

    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    BATCH = 20
    for bi in range(0, len(entries), BATCH):
        batch = entries[bi:bi + BATCH]
        pct = 75 + 20 * (bi + 1) / len(entries)
        tasks[task_id]['progress'] = int(pct)
        tasks[task_id]['message'] = f'百度翻译中... {bi+1}-{min(bi+BATCH, len(entries))}/{len(entries)}'

        texts = '\n'.join(e['text_ja'] for e in batch)
        try:
            salt = '12345'
            sign = hashlib.md5((appid + texts + salt + key).encode('utf-8')).hexdigest()
            resp = _req.post(url, data={
                'q': texts, 'from': 'jp', 'to': 'zh',
                'appid': appid, 'salt': salt, 'sign': sign,
            }, timeout=15).json()
            if 'trans_result' in resp:
                trans = [r['dst'] for r in resp['trans_result']]
            else:
                trans = []
                if bi == 0:
                    tasks[task_id]['message'] = f'百度翻译错误: {str(resp)[:80]}'
        except Exception as ex:
            trans = []
            if bi == 0:
                tasks[task_id]['message'] = f'百度翻译失败: {str(ex)[:80]}'
        for i, e in enumerate(batch):
            e['text_zh'] = trans[i] if i < len(trans) else e['text_ja']


def _translate_deepl(entries, api_key, tasks, task_id):
    """DeepL API 翻译"""
    import requests as _req
    key = api_key or os.environ.get('DEEPL_API_KEY', '')
    if not key:
        for e in entries:
            e['text_zh'] = e['text_ja']
        return

    BATCH = 30
    for bi in range(0, len(entries), BATCH):
        batch = entries[bi:bi + BATCH]
        pct = 75 + 20 * (bi + 1) / len(entries)
        tasks[task_id]['progress'] = int(pct)
        tasks[task_id]['message'] = f'DeepL 翻译中... {bi+1}-{min(bi+BATCH, len(entries))}/{len(entries)}'

        texts = [e['text_ja'] for e in batch]
        try:
            endpoint = 'https://api-free.deepl.com/v2/translate' if key.endswith(':fx') else 'https://api.deepl.com/v2/translate'
            resp = _req.post(endpoint, json={
                'text': texts, 'source_lang': 'JA', 'target_lang': 'ZH',
            }, headers={'Authorization': f'DeepL-Auth-Key {key}'}, timeout=15).json()
            trans = [t['text'] for t in resp['translations']]
        except Exception:
            trans = []
        for i, e in enumerate(batch):
            e['text_zh'] = trans[i] if i < len(trans) else e['text_ja']


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    """启动处理任务"""
    audio = request.files.get('audio')
    if not audio:
        return jsonify({'error': '请上传音频文件'}), 400

    # 保存音频
    ext = Path(audio.filename).suffix or '.mp3'
    audio_path = os.path.join(tempfile.gettempdir(), f'subtitle_{uuid.uuid4().hex}{ext}')
    audio.save(audio_path)

    num_speakers = int(request.form.get('num_speakers', 2))
    max_chars = int(request.form.get('max_chars', 40))
    whisper_model = request.form.get('whisper_model', 'large-v3')
    do_translate = request.form.get('do_translate', 'true') == 'true'
    translator = request.form.get('translator', 'claude')  # claude / deepl
    api_key = request.form.get('api_key', '')

    task_id = uuid.uuid4().hex[:12]
    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'message': '初始化...',
        'files': {},
        'preview': '',
        'stats': '',
    }

    def run():
        tasks[task_id]['message'] = f'设备: {DEVICE}，准备说话人分离...'
        tasks[task_id]['progress'] = 3

        # 等待前一个任务完成（GPU 显存有限，不能并行）
        if not processing_lock.acquire(blocking=False):
            tasks[task_id]['message'] = '排队等待前一个任务完成...'
            tasks[task_id]['progress'] = 2
            processing_lock.acquire()  # 阻塞等待
        try:
            # 步骤1：说话人分离
            tasks[task_id]['message'] = '说话人分离中（约 1 分钟）...'
            tasks[task_id]['progress'] = 5
            diarization = run_diarization(audio_path, device=DEVICE, num_speakers=num_speakers)

            # 步骤2：语音识别
            tasks[task_id]['message'] = '语音识别中...'
            tasks[task_id]['progress'] = 35
            words = run_transcription(audio_path, model_size=whisper_model, device=DEVICE, language='ja', max_chars=max_chars)

            # 步骤3
            tasks[task_id]['message'] = '对齐中...'
            tasks[task_id]['progress'] = 60
            segments = align_speakers_to_transcript(diarization, words, max_chars=max_chars)

            # 步骤4: 翻译
            if do_translate:
                tasks[task_id]['message'] = '翻译中...'
                tasks[task_id]['progress'] = 75
                entries = [{'text_ja': s.text_ja} for s in segments]

                if translator == 'baidu':
                    _translate_baidu(entries, tasks, task_id)
                elif translator == 'google':
                    _translate_google(entries, tasks, task_id)
                elif translator == 'deepl':
                    _translate_deepl(entries, api_key, tasks, task_id)
                else:
                    _translate_claude(entries, api_key, tasks, task_id)

                for i, e in enumerate(entries):
                    segments[i].text_zh = e.get('text_zh', segments[i].text_ja)

            # 生成 SRT 文件
            tasks[task_id]['message'] = '生成 SRT 文件...'
            tasks[task_id]['progress'] = 95

            out_dir = os.path.join(tempfile.gettempdir(), f'subtitle_out_{task_id}')
            os.makedirs(out_dir, exist_ok=True)

            speakers = OrderedDict()
            for s in segments:
                speakers.setdefault(s.speaker, []).append(s)

            for spk, lst in speakers.items():
                fname = f'speaker_{spk.lower()}_ja.srt'
                with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
                    for i, s in enumerate(lst, 1):
                        f.write(f'{i}\n{seconds_to_srt_time(s.start)} --> {seconds_to_srt_time(s.end)}\n{s.text_ja}\n\n')
                tasks[task_id]['files'][fname] = os.path.join(out_dir, fname)

                if do_translate:
                    fname = f'speaker_{spk.lower()}_zh.srt'
                    with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
                        for i, s in enumerate(lst, 1):
                            text = s.text_zh or s.text_ja
                            f.write(f'{i}\n{seconds_to_srt_time(s.start)} --> {seconds_to_srt_time(s.end)}\n{text}\n\n')
                    tasks[task_id]['files'][fname] = os.path.join(out_dir, fname)

            # 预览
            sorted_segs = sorted(segments, key=lambda x: x.start)
            tasks[task_id]['preview'] = '\n'.join(
                f'[{s.speaker}] {s.text_zh or s.text_ja}'
                for s in sorted_segs[:20]
            )
            tasks[task_id]['stats'] = f'共 {len(segments)} 条字幕，{len(speakers)} 个说话人'

            tasks[task_id]['status'] = 'done'
            tasks[task_id]['progress'] = 100
            tasks[task_id]['message'] = '完成！'

        except Exception as e:
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['message'] = str(e)[:500]
        finally:
            processing_lock.release()
            try:
                os.remove(audio_path)
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/api/status/<task_id>')
def status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify({
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
        'files': list(task['files'].keys()),
        'preview': task.get('preview', ''),
        'stats': task.get('stats', ''),
    })


@app.route('/api/download/<task_id>/<filename>')
def download(task_id, filename):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': '任务不存在'}), 404
    filepath = task['files'].get(filename)
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 404
    return send_file(filepath, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
    print('字幕生成器 API 启动 - http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)
