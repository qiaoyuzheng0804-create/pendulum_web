@echo off
cd /d "%~dp0"

:: Stop existing instance
if exist server.pid for /f "usebackq" %%p in ("server.pid") do taskkill /pid %%p /f >nul 2>&1
if exist server.pid del server.pid >nul 2>&1

:: Fallback: if server.pid is missing or stale, kill whatever python still listens on
:: port 5000. Otherwise the new instance dies with "Address already in use" and the
:: shortcut looks like it did nothing at all.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5000 " ^| findstr LISTENING') do (
  tasklist /fi "IMAGENAME eq python.exe" /fi "PID eq %%p" 2>nul | find "python.exe" >nul && taskkill /pid %%p /f >nul 2>&1
)

:: Wait until the port is really free (max 10s) before starting the new instance
set /a wfree=0
:waitfree
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',5000);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel% neq 0 goto :port_free
set /a wfree+=1
if %wfree% geq 10 goto :port_free
ping -n 2 127.0.0.1 >nul
goto :waitfree
:port_free

:: Load environment variables from .env file (skip comments/blank lines)
:: 注意：不能用 "%%a:~0,1%" 这种切片比较——for 变量紧跟的 % 会被 cmd 当作
:: 变量引用展开，把整条 if 撕成非法语法（"命令语法不正确"闪退）。注释行用 eol=# 跳过。
if exist .env for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"

:: Locate Python: prefer the yolov8 conda env (common install locations),
:: then fall back to whatever `python` resolves to on PATH.
set "PYTHON="
if exist "D:\anaconda\envs\yolov8\python.exe" set "PYTHON=D:\anaconda\envs\yolov8\python.exe"
if not defined PYTHON if exist "C:\ProgramData\anaconda3\envs\yolov8\python.exe" set "PYTHON=C:\ProgramData\anaconda3\envs\yolov8\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\anaconda3\envs\yolov8\python.exe" set "PYTHON=%USERPROFILE%\anaconda3\envs\yolov8\python.exe"
if not defined PYTHON set "PYTHON=python"

:: Check if API key is configured (web-saved config llm_config.json also counts,
:: configured later on the web page: AI panel gear button, teacher login required)
:: 注意：提示文字里不能出现未转义的半角括号，括号块内会被当成块结束符导致脚本闪退
if not "%LLM_API_KEY%"=="" goto :key_ok
echo.
echo  [HINT - NOT AN ERROR] LLM_API_KEY not found. The server is starting normally.
echo  Everything works except the AI assistant. To enable AI later you do NOT need
echo  to edit any file: login as teacher, click the gear "Model Config" on the AI
echo  panel, paste your API key and save - it takes effect immediately.
echo.
:key_ok

echo.
echo  Starting server, please wait ...
start "" /min "%PYTHON%" app.py

:: 等待服务端口就绪（最多 60 秒）后立即打开网页，避免模型加载期间打开报错
set /a tries=0
:waitloop
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',5000);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel%==0 goto :open
set /a tries+=1
if %tries% geq 60 goto :timeout
:: ping 延时 1 秒（timeout 在无控制台 stdin 的环境下会报“不支持输入重定向”）
ping -n 2 127.0.0.1 >nul
goto :waitloop
:timeout
echo.
echo  [ERROR] Server did not come up within 60 seconds.
echo  Check the minimized "app.py" window for the Python error, or run this in
echo  the project folder to see the error right here:
echo      "%PYTHON%" app.py
echo.
pause
exit

:open
start "" http://127.0.0.1:5000
exit