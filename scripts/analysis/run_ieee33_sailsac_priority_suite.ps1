param(
    [string]$RepoRoot = "D:\EnergyStorage\Plan\Simple_Microgrid\microgrid_sim",
    [string]$SuitePrefix = "sc33_20k_prio",
    [string]$TrainSteps = "20000",
    [string]$VariantSet = "all"
)

$ErrorActionPreference = 'Stop'

Set-Location $RepoRoot

$dataset = "results/offline_dataset_ieee33_simple_heuristic_seasonal_q124_20260430/combined_offline_dataset.csv"
$losslessOracleRef = "results/oracle_reference_suite_ieee33_simple_2024/oracle_reference_windows.csv"
$networkOracleRef = "results/oracle_reference_suite_ieee33_simple_2024/network_replayed_oracle_reference_windows.csv"
$oracleRef = if (Test-Path $networkOracleRef) { $networkOracleRef } else { $losslessOracleRef }
if ($oracleRef -eq $losslessOracleRef) {
    Write-Warning "Network-replayed oracle reference not found; falling back to lossless LP reference. Run scripts/analysis/build_network_replayed_oracle_reference.py before reviewer-facing runs."
}
$baseArgs = @(
    "scripts/analysis/compare_safe_warmstart_sac.py",
    "--cases","ieee33",
    "--regimes","network_stress",
    "--train-models","simple",
    "--test-models","simple",
    "--agent","sac",
    "--train-steps",$TrainSteps,
    "--seeds","42,52,62",
    "--device","cpu",
    "--offline-dataset",$dataset,
    "--oracle-reference-csv",$oracleRef,
    "--offline-dataset-controller-sources","heuristic_blended_seasonal_elite",
    "--offline-dataset-max-transitions","4000",
    "--shield-delta-penalty-coef","0.1",
    "--online-safe-bc-gradient-steps","32",
    "--online-safe-bc-batch-size","256",
    "--online-safe-bc-max-samples","4000",
    "--online-safe-bc-learning-rate","1e-4",
    "--train-validation-days","30",
    "--train-validation-offset-days-within-year","0,91,182,273",
    "--eval-window-days-list","30,365",
    "--eval-offset-days-list","0,91,182,273",
    "--train-validation-checkpoint-every","250",
    "--train-validation-metric","inventory_value_gate_shield",
    "--train-validation-terminal-penalty-weight","1.0",
    "--train-validation-boundary-dwell-weight","20000",
    "--train-validation-infeasible-dwell-weight","20000",
    "--train-validation-peak-reserve-weight","10000",
    "--train-validation-midband-dwell-weight","20000",
    "--train-validation-soc-target-tracking-weight","10000",
    "--train-validation-peak-discharge-headroom-weight","15000",
    "--train-validation-valley-charge-weight","5000",
    "--train-validation-peak-discharge-weight","5000",
    "--train-validation-gate-dwell-threshold","0.05",
    "--train-validation-shield-mean-delta-threshold","0.05",
    "--train-validation-shield-material-dwell-threshold","0.60",
    "--train-validation-shield-strong-dwell-threshold","0.20",
    "--train-validation-shield-mean-delta-weight","100000",
    "--train-validation-shield-material-dwell-weight","50000",
    "--train-validation-shield-strong-dwell-weight","100000",
    "--train-validation-gate-violation-weight","500000",
    "--train-validation-final-soc-deviation-weight","30000"
)

function Run-Exp([string]$name, [string[]]$extraArgs) {
    $outDir = "results/$name"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "=== Running $name ==="
    & python -u @baseArgs @extraArgs "--output-dir" $outDir
}

function Should-Run([string]$key) {
    $tokens = $VariantSet.Split(",") | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ }
    return ($tokens -contains "all") -or ($tokens -contains $key.ToLowerInvariant())
}

Write-Host "Priority IEEE33 reviewer-facing SAIL-SAC suite"
Write-Host "RepoRoot = $RepoRoot"
Write-Host "SuitePrefix = $SuitePrefix"
Write-Host "TrainSteps = $TrainSteps"
Write-Host "VariantSet = $VariantSet"
Write-Host "Dataset  = $dataset"
Write-Host "Oracle   = $oracleRef"

# P1: strict morphology/value validation-best SAIL-SAC.
if (Should-Run "main") {
    Run-Exp "$($SuitePrefix)_sailsac_maincmp" @(
        "--controller-variants","shielded_replay_warmstart_sac"
    )
}

# P2: no strict morphology/value gate, same fine checkpoint cadence.
if (Should-Run "nostrictgate") {
    Run-Exp "$($SuitePrefix)_sailsac_ablate_nostrictgate" @(
        "--controller-variants","shielded_replay_warmstart_sac",
        "--train-validation-metric","inventory_value",
        "--train-validation-gate-violation-weight","0",
        "--train-validation-shield-mean-delta-weight","0",
        "--train-validation-shield-material-dwell-weight","0",
        "--train-validation-shield-strong-dwell-weight","0",
        "--train-validation-final-soc-deviation-weight","0"
    )
}

# P3: coarse checkpoint cadence with the strict morphology/value gate unchanged.
if (Should-Run "coarseckpt") {
    Run-Exp "$($SuitePrefix)_sailsac_ablate_coarseckpt" @(
        "--controller-variants","shielded_replay_warmstart_sac",
        "--train-validation-checkpoint-every","1000"
    )
}

# P4: balanced continuous value/morphology/shield metric, no hard strict gate.
# Target: recover more value than P1 while avoiding the high-dependence basin of P2.
if (Should-Run "balancedgate") {
    Run-Exp "$($SuitePrefix)_sailsac_balancedgate" @(
        "--controller-variants","shielded_replay_warmstart_sac",
        "--train-validation-checkpoint-every","250",
        "--train-validation-metric","inventory_value_balanced",
        "--train-validation-shield-mean-delta-weight","25000",
        "--train-validation-shield-material-dwell-weight","10000",
        "--train-validation-shield-strong-dwell-weight","25000",
        "--train-validation-final-soc-deviation-weight","10000"
    )
}

Write-Host "=== Priority IEEE33 SAIL-SAC suite completed ==="
