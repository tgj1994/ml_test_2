# Chained watcher:
#   1. Wait for the v5 retrain orchestrator (-V5RetrainPid) to exit.
#   2. Launch the full 30-variant XGB sweep with GPU (XGB_DEVICE=cuda).
#
# XGB writes into preds_cache_xgb/ and logs *_xgb.log, fully separate from the
# EBM cache, so this run does not overwrite anything from the overnight EBM
# sweep. With device='cuda' baked into src/model.py via the XGB_DEVICE env
# var, each XGB fit runs on the RTX 2060 instead of CPU.
#
# Usage:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#       -File runners/post_v5_xgb_sweep.ps1 -V5RetrainPid 31300
param(
    [Parameter(Mandatory = $true)]
    [int]$V5RetrainPid
)

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\jtk\Desktop\coin_prediction_revised'
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir 'post_v5_xgb_sweep.log'
$xgbRunLog = Join-Path $logDir 'run_all_M_sm_xgb_gpu.log'

function Write-Stage($msg) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    "[$ts] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Stage "watcher started, v5RetrainPid=$V5RetrainPid"

# ---- 1. Wait for the v5 retrain orchestrator to exit ----------------------
$proc = Get-Process -Id $V5RetrainPid -ErrorAction SilentlyContinue
if ($proc) {
    Write-Stage "v5 retrain orchestrator is alive; Wait-Process..."
    try {
        Wait-Process -Id $V5RetrainPid -ErrorAction Stop
        Write-Stage "v5 retrain orchestrator exited"
    } catch {
        Write-Stage ("Wait-Process failed: " + $_.Exception.Message)
    }
} else {
    Write-Stage "v5 retrain orchestrator already gone at watcher start; proceeding"
}

# Allow stragglers to flush.
Start-Sleep -Seconds 15

# ---- 2. Launch the full XGB sweep on GPU ----------------------------------
Write-Stage "launching XGB sweep on GPU (XGB_DEVICE=cuda, model=xgb)"
$env:XGB_DEVICE = 'cuda'
Push-Location $root
$xgbProc = Start-Process -FilePath 'uv' `
    -ArgumentList @(
        'run', 'python', 'runners/run_all_M_sm.py',
        '--retrain',
        '--throttle', '5',
        '--threads', '2',
        '--model', 'xgb',
        '--skip-done'
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $xgbRunLog `
    -RedirectStandardError ($xgbRunLog + '.err') `
    -PassThru -WindowStyle Hidden
Write-Stage ("xgb sweep orchestrator pid=" + $xgbProc.Id)
$xgbProc.WaitForExit()
# WaitForExit clears ExitCode if the streams were redirected, mirror the
# success/failure inference via the *_xgb.log Summary line instead.
$summary = (Get-Content $xgbRunLog -Tail 5 -ErrorAction SilentlyContinue |
            Select-String 'Summary:' | Select-Object -Last 1).Line
Pop-Location
Write-Stage ("xgb sweep finished; tail summary: " + $summary)
Write-Stage "watcher done"
