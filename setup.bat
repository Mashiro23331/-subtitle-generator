@echo off
echo ============================
echo   Subtitle Generator Setup
echo ============================
echo.

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] conda not found in PATH.
    echo Please open "Anaconda Prompt" and run this script again.
    echo.
    pause
    exit /b 1
)

echo [1/4] Creating Python environment (2-3 min)...
call conda create -n subtitle python=3.10 -y

echo [2/4] Installing PyTorch + CUDA (5-10 min, large download)...
call conda install -c pytorch -c nvidia pytorch torchaudio pytorch-cuda=12.1 -n subtitle -y

echo [3/4] Installing Python dependencies...
call conda run -n subtitle python -m pip install faster-whisper pyannote.audio anthropic librosa soundfile flask flask-cors deep-translator fugashi unidic-lite

echo [4/4] Done!
echo.
echo ============================
echo   Setup complete!
echo   1. Copy .env.example to .env
echo   2. Fill in your HF_TOKEN
echo   3. Double-click start.bat
echo ============================
pause
