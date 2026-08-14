@echo off
chcp 65001 >nul
echo ============================
echo   安装字幕生成器
echo ============================
echo.

:: 检查 conda 是否可用
where conda >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 conda，请先安装 Anaconda 或 Miniconda
    echo 下载地址: https://www.anaconda.com/download
    pause
    exit /b 1
)

:: 创建 conda 环境
echo [1/4] 创建 Python 环境（约 2-3 分钟）...
call conda create -n subtitle python=3.10 -y

:: 安装 PyTorch（CUDA 12.1）
echo [2/4] 安装 PyTorch + CUDA（约 5-10 分钟，体积大）...
call conda install -c pytorch -c nvidia pytorch torchaudio pytorch-cuda=12.1 -n subtitle -y

:: 安装依赖
echo [3/4] 安装 Python 依赖...
call conda run -n subtitle python -m pip install faster-whisper pyannote.audio anthropic librosa soundfile flask flask-cors deep-translator fugashi unidic-lite

:: 完成
echo [4/4] 完成！
echo.
echo ============================
echo   安装完成！
echo   1. 把 .env.example 复制一份改名为 .env
echo   2. 在 .env 里填入你的 HF_TOKEN
echo   3. 双击 start.bat 启动
echo ============================
pause
