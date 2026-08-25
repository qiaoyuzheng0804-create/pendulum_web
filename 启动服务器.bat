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

:: Locate Python: prefer the yolov8 conda env (common install locations),
:: then fall back to whatever `python` resolves to on PATH.
set "PYTHON="
if exist "D:\anaconda\envs\yolov8\python.exe" set "PYTHON=D:\anaconda\envs\yolov8\python.exe"
if not defined PYTHON if exist "C:\ProgramData\anaconda3\envs\yolov8\python.exe" set "PYTHON=C:\ProgramData\anaconda3\envs\yolov8\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\anaconda3\envs\yolov8\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\envs\yolov8\python.exe"
if not defined PYTHON set "PYTHON=python"

:: Check if API key is configured
if "%LLM_API_KEY%"=="" (
    echo [WARN] LLM_API_KEY not set. AI Q^&A feature will be disabled.
    echo        Edit .env file and set your API key.
)

start "" /min "%PYTHON%" app.py

:: 等待服务端口就绪（最多 60 秒）后立即打开网页，避免模型加载期间打开报错
set /a tries=0
:waitloop
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',5000);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 goto :open
set /a tries+=1
if %tries% geq 60 goto :open
timeout /t 1 /nobreak >nul
goto :waitloop
:open
start "" http://127.0.0.1:5000
exit