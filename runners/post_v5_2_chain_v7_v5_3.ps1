# Chained watcher (runs after the v5_2 watcher finishes):
#   1. Wait for the v5_2 WATCHER (-V5_2WatcherPid) to exit — that watcher
#      itself blocks on XGB then v5_2 sweep, so when it exits both upstream
#      jobs are done. We hop on its tail instead of the sweep PID (which
#      doesn't exist yet at the time we start).
#   2. Launch the v7 EBM sweep (6 variants, ~30-40 min).
#   3. After v7 done, run v5_keep_selector with --cumulative 0.93 to
#      generate src/v5_3_keep.py from v0/v3/v4/v6/v7 results.
#   4. Launch the v5_3 EBM sweep (6 variants, ~15-25 min).
#
# All work runs --retrain so today's expanded FRED + GDELT features are
# fully exercised. CPU only (no GPU), so will not conflict with anything
# else on the box.
param(
    [Parameter(Mandatory = $true)]
    [int]$V5_2WatcherPid
)

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\jtk\Desktop\coin_prediction_revised'
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir 'post_v5_2_chain_v7_v5_3.log'
$v7Log = Join-Path $logDir 'run_v7_sweep.log'
$selectorLog = Join-Path $logDir 'v5_3_selector.log'
$v5_3Log = Join-Path $logDir 'run_v5_3_sweep.log'
$v5_3KeepPy = Join-Path $root 'src\v5_3_keep.py'

function Write-Stage($msg) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    "[$ts] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Stage "watcher started; v5_2 watcher pid=$V5_2WatcherPid"

# ---- 1. Wait for the v5_2 watcher (and therefore XGB + v5_2 sweep) -------
$proc = Get-Process -Id $V5_2WatcherPid -ErrorAction SilentlyContinue
if ($proc) {
    Write-Stage "v5_2 watcher alive; Wait-Process..."
    try { Wait-Process -Id $V5_2WatcherPid -ErrorAction Stop } catch {
        Write-Stage ("Wait-Process failed: " + $_.Exception.Message)
    }
    Write-Stage "v5_2 watcher exited (XGB + v5_2 done)"
} else {
    Write-Stage "v5_2 watcher already gone at watcher start; proceeding"
}
Start-Sleep -Seconds 15

# ---- 2. Launch v7 EBM sweep ----------------------------------------------
Write-Stage "launching v7 EBM sweep (--only-version v7 --retrain)"
Push-Location $root
$proc = Start-Process -FilePath 'uv' `
    -ArgumentList @(
        'run', 'python', 'runners/run_all_M_sm.py',
        '--only-version', 'v7',
        '--retrain',
        '--throttle', '5',
        '--threads', '2',
        '--model', 'ebm'
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $v7Log `
    -RedirectStandardError ($v7Log + '.err') `
    -PassThru -WindowStyle Hidden
Write-Stage ("v7 sweep orchestrator pid=" + $proc.Id)
$proc.WaitForExit()
$v7Summary = (Get-Content $v7Log -Tail 5 -ErrorAction SilentlyContinue |
              Select-String 'Summary:' | Select-Object -Last 1).Line
Pop-Location
Write-Stage ("v7 sweep finished; summary: " + $v7Summary)
Start-Sleep -Seconds 15

# ---- 3. Generate v5_3_keep.py from v7 + non-v5 results -------------------
Write-Stage "launching v5_3 keep selector (--cumulative 0.93)"
$prevMtime = if (Test-Path $v5_3KeepPy) { (Get-Item $v5_3KeepPy).LastWriteTime } else { [DateTime]::MinValue }
Push-Location $root
$proc = Start-Process -FilePath 'uv' `
    -ArgumentList @(
        'run', 'python', 'analysis/v5_keep_selector_ebm.py',
        '--cumulative', '0.93',
        '--output-file', 'src/v5_3_keep.py',
        '--var-name', 'V5_3_KEEP_COLS'
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $selectorLog `
    -RedirectStandardError ($selectorLog + '.err') `
    -PassThru -WindowStyle Hidden
$proc.WaitForExit()
Pop-Location
$keepWritten = (Test-Path $v5_3KeepPy) -and ((Get-Item $v5_3KeepPy).LastWriteTime -gt $prevMtime)
Write-Stage ("v5_3 selector finished; v5_3_keep.py updated: " + $keepWritten)

if (-not $keepWritten) {
    Write-Stage "selector failed to update v5_3_keep.py; ABORTING v5_3 sweep"
    Write-Stage "watcher done"
    exit 1
}

Start-Sleep -Seconds 5

# ---- 4. Launch v5_3 EBM sweep --------------------------------------------
Write-Stage "launching v5_3 EBM sweep (--only-version v5_3 --retrain)"
Push-Location $root
$proc = Start-Process -FilePath 'uv' `
    -ArgumentList @(
        'run', 'python', 'runners/run_all_M_sm.py',
        '--only-version', 'v5_3',
        '--retrain',
        '--throttle', '5',
        '--threads', '2',
        '--model', 'ebm'
    ) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $v5_3Log `
    -RedirectStandardError ($v5_3Log + '.err') `
    -PassThru -WindowStyle Hidden
Write-Stage ("v5_3 sweep orchestrator pid=" + $proc.Id)
$proc.WaitForExit()
$v5_3Summary = (Get-Content $v5_3Log -Tail 5 -ErrorAction SilentlyContinue |
                Select-String 'Summary:' | Select-Object -Last 1).Line
Pop-Location
Write-Stage ("v5_3 sweep finished; summary: " + $v5_3Summary)
Write-Stage "watcher done"
