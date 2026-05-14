# Parallel sweep runner — M + SM cadence variants only.
# Runs `${ThrottleLimit}` variants concurrently. Each variant's EBM uses
# WFConfig.n_jobs=${PerJobThreads} threads (env var). This stays under the
# 12 logical cores on this box while leaving headroom for the OS / bitcoind.
#
# Per-variant logs go to logs/<variant>.log. Failures collected at the end.

param(
    [int] $ThrottleLimit = 4,
    [int] $PerJobThreads = 2,
    [switch] $Retrain
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -Path "$PSScriptRoot\..").Path
Set-Location $ProjectRoot

$logsDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# Variant scripts to run (M + SM cadences only; v0..v6 × {raw, close, complete})
$variants = @()
foreach ($v in @("v0","v3","v4","v5","v6")) {
    foreach ($name in @(
        "main_th_sweep_utc2130.py",
        "main_th_sweep_utc2130_close.py",
        "main_th_sweep_utc2130_complete.py",
        "main_th_sweep_utc2130_sm.py",
        "main_th_sweep_utc2130_sm_close.py",
        "main_th_sweep_utc2130_sm_complete.py"
    )) {
        # v0 omits the _v0 suffix in legacy filenames
        if ($v -ne "v0") { $name = $name.Replace(".py", "_$($v).py") }
        $path = Join-Path $ProjectRoot "main_th_sweep\$v\$name"
        if (Test-Path $path) {
            $variants += [PSCustomObject]@{ Version = $v; Script = $path; Name = $name }
        }
    }
}

Write-Host ("Discovered {0} variants. Throttle={1}, per-job threads={2}." -f $variants.Count, $ThrottleLimit, $PerJobThreads)

# PowerShell 5.1: use Start-Job and throttle manually
$running = @{}
$completed = @()
$failed = @()
$variantQueue = [System.Collections.Generic.Queue[object]]::new()
foreach ($v in $variants) { $variantQueue.Enqueue($v) }

while ($variantQueue.Count -gt 0 -or $running.Count -gt 0) {
    while ($running.Count -lt $ThrottleLimit -and $variantQueue.Count -gt 0) {
        $v = $variantQueue.Dequeue()
        $logFile = Join-Path $logsDir ("{0}_{1}.log" -f $v.Version, [System.IO.Path]::GetFileNameWithoutExtension($v.Name))
        $retrainArg = if ($Retrain.IsPresent) { "--retrain" } else { "" }
        $env:EBM_N_JOBS = "$PerJobThreads"
        $env:PYTHONIOENCODING = "utf-8"
        # If $env:MODEL_KIND already set by the caller (e.g. 'xgb'), inherit it.
        $proc = Start-Process -FilePath "uv" `
            -ArgumentList @("run","python", $v.Script, $retrainArg) `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $logFile `
            -RedirectStandardError ($logFile + ".err") `
            -NoNewWindow -PassThru
        $running[$proc.Id] = [PSCustomObject]@{ Process = $proc; Variant = $v; LogFile = $logFile; Started = Get-Date }
        Write-Host ("  START [pid={0}] {1}/{2}" -f $proc.Id, $v.Version, $v.Name)
    }
    Start-Sleep -Seconds 5
    foreach ($key in @($running.Keys)) {
        $job = $running[$key]
        if ($job.Process.HasExited) {
            $elapsed = (Get-Date) - $job.Started
            if ($job.Process.ExitCode -eq 0) {
                Write-Host ("  DONE  [{0:N0}s] {1}/{2}" -f $elapsed.TotalSeconds, $job.Variant.Version, $job.Variant.Name)
                $completed += $job
            } else {
                Write-Host ("  FAIL  [exit={0}] {1}/{2}  -> {3}" -f $job.Process.ExitCode, $job.Variant.Version, $job.Variant.Name, $job.LogFile)
                $failed += $job
            }
            $running.Remove($key)
        }
    }
}

Write-Host ("`nSummary: completed={0}, failed={1}" -f $completed.Count, $failed.Count)
if ($failed.Count -gt 0) {
    Write-Host "Failures:"
    $failed | ForEach-Object { Write-Host ("  {0}/{1}  log: {2}" -f $_.Variant.Version, $_.Variant.Name, $_.LogFile) }
    exit 1
}
