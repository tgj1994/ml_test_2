# Full post-EBM chain:
#   1. Wait for the EBM orchestrator (-EbmPid) to exit.
#   2. Run analysis/v5_keep_selector_ebm.py to regenerate src/v5_keep.py from
#      the fresh non-v5 EBM importance CSVs.
#   3. If selector wrote a v5_keep.py, re-run only the v5 variants with
#      --retrain so they pick up the new keep list.
#   4. Run the full 30-variant XGB sweep on GPU for the apples-to-apples
#      EBM-vs-XGB comparison the user requested.
#
# Each major step writes a stage line into logs/post_ebm_full_chain.log so
# progress is observable after a reboot. Step return codes from
# Start-Process + redirected streams are unreliable (the parent often gets
# null), so we infer step success from artifacts on disk (v5_keep.py
# existence, *_xgb.log Summary line) instead of $proc.ExitCode.
#
# Usage:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass `
#       -File runners/post_ebm_full_chain.ps1 -EbmPid 12164
param(
    [Parameter(Mandatory = $true)]
    [int]$EbmPid
)

$ErrorActionPreference = 'Continue'
$root = 'C:\Users\jtk\Desktop\coin_prediction_revised'
$logDir = Join-Path $root 'logs'
$logFile = Join-Path $logDir 'post_ebm_full_chain.log'
$selectorLog = Join-Path $logDir 'v5_keep_selector_v2.log'
$v5RunLog = Join-Path $logDir 'run_v5_retrain_v2.log'
$xgbRunLog = Join-Path $logDir 'run_all_M_sm_xgb_gpu_v2.log'
$v5KeepPy = Join-Path $root 'src\v5_keep.py'

function Write-Stage($msg) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    "[$ts] $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Stage "chain watcher started; ebmPid=$EbmPid"

# ---- 1. Wait for the EBM orchestrator to exit -----------------------------
$proc = Get-Process -Id $EbmPid -ErrorAction SilentlyContinue
if ($proc) {
    Write-Stage "EBM orchestrator alive; Wait-Process..."
    try { Wait-Process -Id $EbmPid -ErrorAction Stop } catch {
        Write-Stage ("Wait-Process failed: " + $_.Exception.Message)
    }
    Write-Stage "EBM orchestrator exited"
} else {
    Write-Stage "EBM orchestrator already gone at start; proceeding"
}
Start-Sleep -Seconds 15

# ---- 2. Run v5 keep selector ----------------------------------------------
$prevV5KeepMtime = if (Test-Path $v5KeepPy) { (Get-Item $v5KeepPy).LastWriteTime } else { [DateTime]::MinValue }
Write-Stage "launching v5_keep_selector_ebm.py"
Push-Location $root
$selectorProc = Start-Process -FilePath 'uv' `
    -ArgumentList @('run', 'python', 'analysis/v5_keep_selector_ebm.py') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $selectorLog `
    -RedirectStandardError ($selectorLog + '.err') `
    -PassThru -WindowStyle Hidden
$selectorProc.WaitForExit()
Pop-Location
$newKeepWritten = (Test-Path $v5KeepPy) -and ((Get-Item $v5KeepPy).LastWriteTime -gt $prevV5KeepMtime)
Write-Stage ("selector finished; v5_keep.py updated: " + $newKeepWritten)

# ---- 3. If selector produced a new keep, re-run v5 variants ---------------
if ($newKeepWritten) {
    Write-Stage "launching v5 retrain (--only-version v5 --retrain)"
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
    Pop-Location
    $v5Summary = (Get-Content $v5RunLog -Tail 5 -ErrorAction SilentlyContinue |
                  Select-String 'Summary:' | Select-Object -Last 1).Line
    Write-Stage ("v5 retrain finished; summary: " + $v5Summary)
} else {
    Write-Stage "skipping v5 retrain (no new keep)"
}
Start-Sleep -Seconds 5

# NB: utc2130_runner.py now tags every output (threshold_sweep_*,
# label_threshold_sweep_summary_*, label_prob_threshold_tables_*,
# equity_curve_*, label_x_prob_*_heatmap_*) with `{model_kind}` so EBM and
# XGB no longer share output filenames. The earlier snapshot step that
# copied unsuffixed -> *_ebm has been removed because there is nothing
# unsuffixed left to copy.

# ---- 4. Run the full XGB sweep on GPU -------------------------------------
Write-Stage "launching full XGB GPU sweep (--retrain --model xgb, XGB_DEVICE=cuda)"
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
$xgbSummary = (Get-Content $xgbRunLog -Tail 5 -ErrorAction SilentlyContinue |
               Select-String 'Summary:' | Select-Object -Last 1).Line
Pop-Location
Write-Stage ("xgb sweep finished; summary: " + $xgbSummary)

# Same reasoning as the removed step 3b: utc2130_runner.py now writes
# `_xgb`-tagged paths natively, so no post-XGB snapshot is needed.
Write-Stage "chain watcher done"
