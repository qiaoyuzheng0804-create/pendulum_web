@echo off
cd /d "%~dp0"

if not exist server.pid (
    echo Server not running.
    pause
    exit
)

:: Step 1: graceful stop - tell the watchdog the page closed.
:: This runs the normal shutdown path (electromagnet power-off + serial cleanup),
:: which taskkill /f would skip. Server exits by itself within ~15s.
powershell -NoProfile -Command "try{Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/client_exit' -Method POST -ContentType 'application/json' -Body '{\"id\":\"manual-stop\"}' -TimeoutSec 3 | Out-Null}catch{}" >nul 2>&1

echo Stopping server (graceful, up to 20s)...
set /a tries=0
:waitloop
powershell -NoProfile -Command "if (Get-Process -Id (Get-Content server.pid -ErrorAction SilentlyContinue) -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" >nul 2>&1
if %errorlevel%==0 goto :stopped
set /a tries+=1
if %tries% geq 20 goto :force
timeout /t 1 /nobreak >nul
goto :waitloop

:: Step 2: fallback - force kill if still alive
:force
set /p PID=<server.pid
taskkill /pid %PID% /f >nul 2>&1
echo Server force-stopped (graceful stop timed out).

:stopped
del server.pid >nul 2>&1
echo Server stopped.
timeout /t 2 /nobreak >nul
exit
