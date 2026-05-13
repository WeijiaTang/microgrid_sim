param(
    [string]$RepoRoot = "D:\EnergyStorage\Plan\Simple_Microgrid\microgrid_sim",
    [string]$WatchOutputDir = "results/sc33_15k3s_srw_q30_strictg500",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = 'Stop'

Set-Location $RepoRoot

$watchSummary = Join-Path $RepoRoot (Join-Path $WatchOutputDir "summary.csv")
$watchGrouped = Join-Path $RepoRoot (Join-Path $WatchOutputDir "summary_grouped.csv")
$reportPath = Join-Path $RepoRoot (Join-Path $WatchOutputDir "autopilot_report.txt")
$nextOutDir = "results/sc33_20k3s_srw_q30_strictg250"

function Write-Report([string]$line) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $payload = "[$ts] $line"
    Add-Content -Path $reportPath -Value $payload -Encoding UTF8
    Write-Output $payload
}

function Get-GroupedRow([string]$path) {
    if (-not (Test-Path $path)) { return $null }
    $rows = Import-Csv $path
    if ($rows.Count -lt 1) { return $null }
    return $rows[0]
}

Write-Report "Autopilot watcher started. Watching $WatchOutputDir"

while (-not (Test-Path $watchSummary)) {
    $job = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*$($WatchOutputDir.Replace('\', '/'))*" -or
        $_.CommandLine -like "*$($WatchOutputDir.Replace('/', '\'))*"
    } | Where-Object { $_.Name -eq "python.exe" } | Select-Object -First 1

    if ($job) {
        $proc = Get-Process -Id $job.ProcessId -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Report "Still running pid=$($job.ProcessId) cpu=$([math]::Round($proc.CPU, 2)) wsMB=$([math]::Round($proc.WorkingSet64 / 1MB, 1))"
        } else {
            Write-Report "Process metadata found but Get-Process failed for pid=$($job.ProcessId)"
        }
    } else {
        Write-Report "No active python process found for $WatchOutputDir yet summary.csv not present; continuing to wait."
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-Report "Detected summary.csv for $WatchOutputDir"

$current = Get-GroupedRow $watchGrouped
$base5 = Get-GroupedRow (Join-Path $RepoRoot "results/sc33_5k3s_safev1/summary_grouped.csv")
$base7 = Get-GroupedRow (Join-Path $RepoRoot "results/sc33_10k3s_srw_valshield/summary_grouped.csv")
$base30 = Get-GroupedRow (Join-Path $RepoRoot "results/sc33_10k3s_srw_valshield_q30/summary_grouped.csv")

if (-not $current) {
    Write-Report "Current grouped summary missing or empty. Exiting."
    exit 1
}

$curObjective = [double]$current.mean_final_cumulative_objective_cost
$curDelta = [double]$current.mean_abs_shield_delta
$curMaterial = [double]$current.mean_shield_material_activation_fraction
$curStrong = [double]$current.mean_shield_strong_activation_fraction

Write-Report ("Current grouped metrics: objective={0:F6}, mean_abs_shield_delta={1:F6}, material={2:F6}, strong={3:F6}" -f $curObjective, $curDelta, $curMaterial, $curStrong)

if ($base5) {
    Write-Report ("Delta vs 5k_safev1 objective = {0:F6}" -f ($curObjective - [double]$base5.mean_final_cumulative_objective_cost))
}
if ($base7) {
    Write-Report ("Delta vs 10k_valshield objective = {0:F6}" -f ($curObjective - [double]$base7.mean_final_cumulative_objective_cost))
}
if ($base30) {
    Write-Report ("Delta vs 10k_valshield_q30 objective = {0:F6}" -f ($curObjective - [double]$base30.mean_final_cumulative_objective_cost))
}

$needNext = $false
if ($curObjective -gt 907686.64) { $needNext = $true; Write-Report "Objective still above no-storage reference 907686.64" }
if ($curDelta -gt 0.01) { $needNext = $true; Write-Report "mean_abs_shield_delta still above 0.01" }
if ($curMaterial -gt 0.20) { $needNext = $true; Write-Report "shield_material_activation_fraction still above 0.20" }
if ($curStrong -gt 0.02) { $needNext = $true; Write-Report "shield_strong_activation_fraction still above 0.02" }

if (-not $needNext) {
    Write-Report "Current run meets strict autopilot target; no next run launched."
    exit 0
}

if (Test-Path (Join-Path $RepoRoot $nextOutDir)) {
    Write-Report "Next output dir $nextOutDir already exists. Refusing to auto-launch duplicate run."
    exit 0
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $nextOutDir) | Out-Null
$stdout = Join-Path $RepoRoot (Join-Path $nextOutDir "run.out.log")
$stderr = Join-Path $RepoRoot (Join-Path $nextOutDir "run.err.log")

$args = @(
    "scripts/analysis/compare_safe_warmstart_sac.py",
    "--cases","ieee33",
    "--regimes","network_stress",
    "--train-models","simple",
    "--test-models","simple",
    "--controller-variants","shielded_replay_warmstart_sac",
    "--agent","sac",
    "--train-steps","20000",
    "--seeds","42,52,62",
    "--device","cpu",
    "--offline-dataset","results/offline_dataset_ieee33_simple_heuristic_seasonal_q124_20260430/combined_offline_dataset.csv",
    "--offline-dataset-controller-sources","heuristic_blended_seasonal_elite",
    "--offline-dataset-max-transitions","4000",
    "--shield-delta-penalty-coef","0.1",
    "--online-safe-bc-gradient-steps","32",
    "--online-safe-bc-batch-size","256",
    "--online-safe-bc-max-samples","4000",
    "--online-safe-bc-learning-rate","1e-4",
    "--train-validation-days","30",
    "--train-validation-offset-days-within-year","0,91,182,273",
    "--train-validation-checkpoint-every","250",
    "--train-validation-metric","health_objective_gate_shield",
    "--train-validation-terminal-penalty-weight","1.0",
    "--train-validation-boundary-dwell-weight","20000",
    "--train-validation-infeasible-dwell-weight","20000",
    "--train-validation-peak-reserve-weight","10000",
    "--train-validation-gate-dwell-threshold","0.03",
    "--train-validation-gate-violation-weight","500000",
    "--train-validation-shield-mean-delta-weight","100000",
    "--train-validation-shield-material-dwell-weight","50000",
    "--train-validation-shield-strong-dwell-weight","100000",
    "--train-validation-final-soc-deviation-weight","30000",
    "--train-validation-shield-mean-delta-threshold","0.01",
    "--train-validation-shield-material-dwell-threshold","0.20",
    "--train-validation-shield-strong-dwell-threshold","0.02",
    "--output-dir",$nextOutDir
)

$proc = Start-Process -FilePath python -ArgumentList $args -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Write-Report "Auto-launched next run $nextOutDir with pid=$($proc.Id)"
