# 🎙️ 日语字幕生成器

上传音频 → 自动识别说话人 → 日语转写 → 中文翻译 → 下载 SRT 字幕

## 功能

- ✅ 说话人分离（pyannote-audio，自动区分不同人）
- ✅ 语音识别（faster-whisper large-v3，日语转写）
- ✅ 中文翻译（Claude / Google / DeepL 可选）
- ✅ 字幕生成（按说话人分离 + 合并版，标准 SRT 格式）
- ✅ Web 界面（浏览器操作，拖拽上传）

## 环境要求

- **NVIDIA GPU**（8GB 显存以上，推荐 RTX 3060 及以上）
- **Windows 10/11**
- **Anaconda**（用于创建隔离环境）

## 安装步骤

1. 安装 [Anaconda](https://www.anaconda.com/download)

2. 双击 `setup.bat`，等待安装完成（约 10 分钟）

3. 配置 Token：
   - 复制 `.env.example` 改名为 `.env`
   - 填入你的 HuggingFace Token（免费注册 https://huggingface.co 获取）

4. 双击 `start.bat` 启动

5. 浏览器打开 http://127.0.0.1:5000

## 使用

1. 拖拽音频文件到上传区（支持 MP3/WAV/M4A/FLAC）
2. 设置参数：
   - **说话人数**：音频里有几个人说话
   - **最大字数**：单条字幕最长多少字
   - **Whisper 模型**：large-v3 最准，medium 更快
   - **翻译引擎**：Claude（最准）/ Google（免费）/ DeepL
3. 点击"开始处理"
4. 等待 3-5 分钟
5. 下载生成的 SRT 文件

## 获取 HuggingFace Token

1. 注册 https://huggingface.co
2. 访问 Settings → Access Tokens → New token
3. 访问以下两个页面，点击 "Agree and access repository"：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
4. 把 Token 填入 `.env` 文件

## 文件说明

| 文件 | 用途 |
|------|------|
| `server.py` | Flask 后端 API |
| `index.html` | Web 前端页面 |
| `generate_subtitles.py` | 核心处理 pipeline |
| `start.bat` | 启动脚本 |
| `setup.bat` | 安装脚本 |
| `.env.example` | Token 配置模板 |

## 常见问题

**Q: 进度条不动？**
A: 说话人分离和语音识别各需 1-2 分钟，进度条有呼吸动画表示仍在工作。

**Q: 报 CUDA out of memory？**
A: 显存不足，改用 `medium` 模型，或关掉其他占用 GPU 的程序。

**Q: Google 翻译不可用？**
A: Google 翻译在国内可能被墙，改用 Claude（默认）或 DeepL。
