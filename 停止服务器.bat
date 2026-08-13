@echo off
cd /d "%~dp0"

if not exist server.pid (
    echo Server not running.
    pause
    exit
)

set /p PID=<server.pid
taskkill /pid %PID% /f >nul 2>&1
del server.pid >nul 2>&1

echo Server stopped.
timeout /t 2 /nobreak >nul
exit
