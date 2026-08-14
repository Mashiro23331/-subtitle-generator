@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: 从 .env 文件加载 Token（该文件不上传 GitHub）
if exist .env for /f "tokens=*" %%a in (.env) do set %%a

:: 检查 .env 是否存在
if not exist .env (
    echo [提示] 未找到 .env 文件
    echo   请复制 .env.example 为 .env 并填入 HF_TOKEN
    echo.
)

:: 启动
echo ============================
echo   日语字幕生成器
echo   浏览器打开 http://127.0.0.1:5000
echo ============================

conda run -n subtitle python server.py
pause
