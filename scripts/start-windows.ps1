$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RootDir

$Port = if ($env:MANHUA_PORT) { $env:MANHUA_PORT } elseif ($env:PORT) { $env:PORT } else { "8002" }

$ParsedPort = 0
if (-not [int]::TryParse($Port, [ref]$ParsedPort) -or $ParsedPort -lt 1 -or $ParsedPort -gt 65535) {
  Write-Error "MANHUA_PORT/PORT must be an integer between 1 and 65535."
  exit 2
}
$Port = [string]$ParsedPort

if (-not (Test-Path ".venv")) {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv .venv
  } else {
    python -m venv .venv
  }
}

$Python = Join-Path $RootDir ".venv\Scripts\python.exe"
& $Python -m pip install -r requirements.txt

# 停止旧实例（先优雅关闭，再强杀）
$oldConnections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($oldConnections) {
    $oldPids = $oldConnections.OwningProcess | Sort-Object -Unique
    foreach ($p in $oldPids) {
        if ($p -gt 0) {
            Write-Host "正在关闭端口 $Port 上的旧进程 (PID $p)..."
            Stop-Process -Id $p -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep 5
    $remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($remaining) {
        $remaining | ForEach-Object {
            if ($_.OwningProcess -gt 0) {
                Write-Host "强制终止残留进程 (PID $($_.OwningProcess))"
                Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep 2
    }
}

$env:MANHUA_PORT = $Port
$Url = "http://localhost:$Port"
$HealthUrl = "$Url/api/health"
Start-Job -ScriptBlock {
  param($HealthUrl, $Url)
  for ($i = 0; $i -lt 60; $i++) {
    try {
      Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 1 | Out-Null
      Start-Process $Url
      return
    } catch {
      Start-Sleep -Seconds 1
    }
  }
} -ArgumentList $HealthUrl, $Url | Out-Null
& $Python main.py
