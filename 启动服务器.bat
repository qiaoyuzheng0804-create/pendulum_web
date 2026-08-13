@echo off
cd /d "%~dp0"

:: Stop existing instance
if exist server.pid (
    set /p PID=<server.pid
    taskkill /pid %PID% /f >nul 2>&1
    del server.pid >nul 2>&1
)

:: Load environment variables from .env file
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

:: Start server with yolov8 conda environment
set PYTHON=D:\anaconda\envs\yolov8\python.exe

:: Check if API key is configured
if "%MIMO_API_KEY%"=="" (
    echo [WARN] MIMO_API_KEY not set. AI Q^&A feature will be disabled.
    echo        Edit .env file and set your API key.
)

start "" /min "%PYTHON%" app.py

:: Open browser after a short delay
timeout /t 5 /nobreak >nul
start "" http://127.0.0.1:5000