# 启动 narration-workflow 开发服务器
# 用法: powershell -ExecutionPolicy Bypass -File start.ps1

$PORT = 8003

# 用 Get-NetTCPConnection 精确获取 LISTENING 状态的 PID
$connections = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
if ($connections) {
    $owningPids = $connections.OwningProcess | Sort-Object -Unique
    # 先尝试优雅关闭（发送 WM_CLOSE，让 uvicorn 走 shutdown 流程）
    foreach ($p in $owningPids) {
        if ($p -gt 0) {
            Write-Host "正在关闭端口 $PORT 上的进程 (PID $p)..."
            Stop-Process -Id $p -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep 5
    # 仍在运行的进程，强制终止
    $remaining = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    if ($remaining) {
        $remainingPids = $remaining.OwningProcess | Sort-Object -Unique
        foreach ($p in $remainingPids) {
            if ($p -gt 0) {
                Write-Host "强制终止残留进程 (PID $p)"
                Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep 2
    }
    # 最终确认
    $final = Get-NetTCPConnection -LocalPort $PORT -State Listen -ErrorAction SilentlyContinue
    if ($final) {
        Write-Host "警告：端口 $PORT 仍被占用，可能需要手动处理"
    }
}

# 清理 Python 字节码缓存，确保加载最新代码
Get-ChildItem -Path $PSScriptRoot -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 启动服务器
Write-Host "启动服务器 http://localhost:$PORT"
Write-Host "[提示] 停止服务器请按 Ctrl+C，不要直接关闭窗口！" -ForegroundColor Yellow
Set-Location $PSScriptRoot
python main.py
