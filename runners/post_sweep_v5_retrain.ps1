# Post-sweep automation:
#   1. Wait for the running orchestrator (passed as -OrchestratorPid) to exit.
#   2. Run analysis/v5_keep_selector_ebm.py which writes src/v5_keep.py.
#   3. If src/v5_keep.py was written, re-run only the v5 variants with --retrain.
#
# Launched in the background by the main session. Logs every step to
# logs/post_sweep_v5_retrain.log so progress is observable after a reboot or
# session disconnect.
#
# Usage:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#       -File runners/post_sweep_v5_retrain.ps1 -OrchestratorPid 19524
param(
    [Parameter(Mandatory = $true)]
    [int]$OrchestratorPid
)

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\jtk\Desktop\coin_prediction_revised'
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir 'post_sweep_v5_retrain.log'
$selectorLog = Join-Path $logDir 'v5_keep_selector.log'
$v5RunLog = Join-Path $logDir 'run_v5_retrain.log'
$v5KeepPy = Join-Path $root 'src\v5_keep.py'

function Write-Stage($msg) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    "[$ts] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Stage "watcher started, orchestratorPid=$OrchestratorPid"

# ---- 1. Wait for orchestrator to exit -------------------------------------
$proc = Get-Process -Id $OrchestratorPid -ErrorAction SilentlyContinue
if ($proc) {
    Write-Stage "orchestrator is alive; Wait-Process..."
    try {
        Wait-Process -Id $OrchestratorPid -ErrorAction Stop
        Write-Stage "orchestrator exited"
    } catch {
        Write-Stage ("Wait-Process failed: " + $_.Exception.Message)
    }
} else {
    Write-Stage "orchestrator already gone at watcher start; proceeding"
}

# Give the OS a moment to flush all child sweep processes' final report writes.
Start-Sleep -Seconds 15

# ---- 2. Sanity-check inputs for the v5 selector ---------------------------
$fiCsvs = Get-ChildItem -Path (Join-Path $root 'reports') -Recurse `
    -Filter 'feature_importance_ebm_*.csv' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike '*_v5_*' -and $_.FullName -notlike '*\v5_*' }
Write-Stage ("feature_importance_ebm_*.csv count (excl. v5): " + $fiCsvs.Count)
if ($fiCsvs.Count -lt 50) {
    Write-Stage "WARNING: fewer than 50 fi CSVs found; selector may produce a weak keep list, but proceeding anyway"
}

# ---- 3. Run the v5 selector -----------------------------------------------
Write-Stage "launching v5_keep_selector_ebm.py"
Push-Location $root
$selectorProc = Start-Process -FilePath 'uv' `
    -ArgumentList @('run', 'python', 'analysis/v5_keep_selector_ebm.py') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $selectorLog `
    -RedirectStandardError ($selectorLog + '.err') `
    -PassThru -WindowStyle Hidden
$selectorProc.WaitForExit()
$selectorRC = $selectorProc.ExitCode
Pop-Location
Write-Stage ("selector exit code: $selectorRC")

if ($selectorRC -ne 0 -or -not (Test-Path $v5KeepPy)) {
    Write-Stage "selector failed or v5_keep.py missing; ABORTING v5 retrain"
    exit 1
}
Write-Stage ("src/v5_keep.py written, size=" + (Get-Item $v5KeepPy).Length + "B")

# ---- 4. Re-run only the v5 variants with --retrain ------------------------
Write-Stage "launching v5 retrain (run_all_M_sm.py --only-version v5 --retrain)"
Push-Location $root
$v5Proc = Start-Process -FilePath 'uv' `
    -ArgumentList @(
        'run', 'python', 'runners/run_all_M_sm.py',
        '--only-version', 'v5',
        '--retrain',
        '--throttle', '5',
        '--threads', '2',
        '--model', 'ebm'
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $v5RunLog `
    -RedirectStandardError ($v5RunLog + '.err') `
    -PassThru -WindowStyle Hidden
$v5Proc.WaitForExit()
$v5RC = $v5Proc.ExitCode
Pop-Location
Write-Stage ("v5 retrain orchestrator exit code: $v5RC")

if ($v5RC -eq 0) {
    Write-Stage "v5 retrain completed successfully"
} else {
    Write-Stage "v5 retrain finished with errors (rc=$v5RC)"
}
Write-Stage "watcher done"
