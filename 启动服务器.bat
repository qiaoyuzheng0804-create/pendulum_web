@echo off
cd /d "%~dp0"

:: Stop existing instance
if exist server.pid (
    set /p PID=<server.pid
    taskkill /pid %PID% /f >nul 2>&1
    del server.pid >nul 2>&1
)

:: Start server with yolov8 conda environment
set PYTHON=D:\anaconda\envs\yolov8\python.exe

:: MIMO AI API - set your own key below (do NOT commit the real key to Git)
set MIMO_API_KEY=
if "%MIMO_API_KEY%"=="" (
    echo [WARN] MIMO_API_KEY not set. AI Q&A feature will be disabled.
    echo        Edit this file and put your key after "set MIMO_API_KEY=".
)
set MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
set MIMO_MODEL=mimo-v2.5

start "" /min "%PYTHON%" app.py

:: Open browser after a short delay
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:5000
