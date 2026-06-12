@echo off
chcp 65001 >nul
title 解说漫工作台

echo 正在停止旧实例...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8003 ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a >nul 2>&1
)
timeout /t 3 /nobreak >nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8003 ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo 启动服务器...
echo [提示] 停止服务器请按 Ctrl+C，不要直接关闭窗口！
cd /d "%~dp0"
start "" "http://localhost:8003"
python main.py
pause
