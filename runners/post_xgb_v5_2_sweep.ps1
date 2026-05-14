# Chained watcher:
#   1. Wait for the current XGB CPU sweep orchestrator (-XgbPid) to exit.
#   2. Launch the v5_2 EBM sweep — only the 6 v5_2 variants with --retrain.
#
# Why this exists: the user opened a "loosened keep set" experiment after
# the main EBM/XGB pipelines were already in flight. We want the v5_2
# sweep to run on a fully idle box (XGB is CPU-only and at 99% saturation)
# so it doesn't fight for cycles, and to start the moment XGB is done.
#
# Usage:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#       -File runners/post_xgb_v5_2_sweep.ps1 -XgbPid 19700
param(
    [Parameter(Mandatory = $true)]
    [int]$XgbPid
)

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\jtk\Desktop\coin_prediction_revised'
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir 'post_xgb_v5_2_sweep.log'
$v52RunLog = Join-Path $logDir 'run_v5_2_sweep.log'

function Write-Stage($msg) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    "[$ts] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Stage "watcher started, xgbPid=$XgbPid"

# ---- Wait for XGB to exit ------------------------------------------------
$proc = Get-Process -Id $XgbPid -ErrorAction SilentlyContinue
if ($proc) {
    Write-Stage "XGB orchestrator alive; Wait-Process..."
    try {
        Wait-Process -Id $XgbPid -ErrorAction Stop
        Write-Stage "XGB orchestrator exited"
    } catch {
        Write-Stage ("Wait-Process failed: " + $_.Exception.Message)
    }
} else {
    Write-Stage "XGB orchestrator already gone at watcher start; proceeding"
}

Start-Sleep -Seconds 15

# ---- Launch v5_2 sweep ---------------------------------------------------
Write-Stage "launching v5_2 EBM sweep (--only-version v5_2 --retrain)"
Push-Location $root
$proc = Start-Process -FilePath 'uv' `
    -ArgumentList @(
        'run', 'python', 'runners/run_all_M_sm.py',
        '--only-version', 'v5_2',
        '--retrain',
        '--throttle', '5',
        '--threads', '2',
        '--model', 'ebm'
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $v52RunLog `
    -RedirectStandardError ($v52RunLog + '.err') `
    -PassThru -WindowStyle Hidden
Write-Stage ("v5_2 sweep orchestrator pid=" + $proc.Id)
$proc.WaitForExit()
$summary = (Get-Content $v52RunLog -Tail 5 -ErrorAction SilentlyContinue |
            Select-String 'Summary:' | Select-Object -Last 1).Line
Pop-Location
Write-Stage ("v5_2 sweep finished; summary: " + $summary)
Write-Stage "watcher done"
