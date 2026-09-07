@echo off
cd /d "%~dp0"

if exist .env for /f "tokens=*" %%a in (.env) do set %%a

if not exist .env (
    echo [WARNING] .env not found. Copy .env.example to .env and fill HF_TOKEN.
    echo.
)

echo ============================
echo   Subtitle Generator
echo   Open http://127.0.0.1:5000
echo ============================

rem Try common conda locations
set PY=
if exist "D:\anaconda3\envs\pytorch\python.exe" set PY=D:\anaconda3\envs\pytorch\python.exe
if exist "C:\Users\%USERNAME%\anaconda3\envs\pytorch\python.exe" set PY=C:\Users\%USERNAME%\anaconda3\envs\pytorch\python.exe
if exist "C:\Users\%USERNAME%\miniconda3\envs\pytorch\python.exe" set PY=C:\Users\%USERNAME%\miniconda3\envs\pytorch\python.exe

if "%PY%"=="" (
    echo [ERROR] Python environment not found.
    echo   Option 1: Run setup.bat first to install.
    echo   Option 2: Edit start.bat and set PY to your python path.
    echo.
    pause
    exit /b 1
)

"%PY%" server.py
pause
