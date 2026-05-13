param(
    [string]$RepoRoot = "D:\EnergyStorage\Plan\Simple_Microgrid\microgrid_sim"
)

$ErrorActionPreference = 'Stop'

Set-Location $RepoRoot

$dataset = "results/offline_dataset_ieee33_simple_heuristic_seasonal_q124_20260430/combined_offline_dataset.csv"
$oracleRef = "results/oracle_reference_suite_ieee33_simple_2024/oracle_reference_windows.csv"
$baseArgs = @(
    "scripts/analysis/compare_safe_warmstart_sac.py",
    "--cases","ieee33",
    "--regimes","network_stress",
    "--train-models","simple",
    "--test-models","simple",
    "--agent","sac",
    "--train-steps","20000",
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
    "--train-validation-metric","inventory_value_gate",
    "--train-validation-terminal-penalty-weight","1.0",
    "--train-validation-boundary-dwell-weight","20000",
    "--train-validation-infeasible-dwell-weight","20000",
    "--train-validation-peak-reserve-weight","10000",
    "--train-validation-midband-dwell-weight","20000",
    "--train-validation-soc-target-tracking-weight","10000",
    "--train-validation-peak-discharge-headroom-weight","15000",
    "--train-validation-valley-charge-weight","5000",
    "--train-validation-peak-discharge-weight","5000",
    "--train-validation-gate-dwell-threshold","0.03",
    "--train-validation-gate-violation-weight","500000",
    "--train-validation-final-soc-deviation-weight","30000"
)

function Run-Exp([string]$name, [string[]]$extraArgs) {
    $outDir = "results/$name"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "=== Running $name ==="
    & python @baseArgs @extraArgs "--output-dir" $outDir
}

Write-Host "Final IEEE33 SAIL-SAC suite"
Write-Host "RepoRoot = $RepoRoot"
Write-Host "Dataset  = $dataset"
Write-Host "Oracle   = $oracleRef"

# E1: main comparison rows
Run-Exp "sc33_20k3s_plain_maincmp" @(
    "--controller-variants","plain_sac"
)

Run-Exp "sc33_20k3s_shielded_maincmp" @(
    "--controller-variants","shielded_sac"
)

Run-Exp "sc33_20k3s_sailsac_maincmp" @(
    "--controller-variants","shielded_replay_warmstart_sac"
)

# E3: core ablation rows
Run-Exp "sc33_20k3s_sailsac_ablate_nosafebc" @(
    "--controller-variants","shielded_replay_warmstart_sac",
    "--online-safe-bc-gradient-steps","0"
)

Run-Exp "sc33_20k3s_sailsac_ablate_noreplaywarm" @(
    "--controller-variants","shielded_sac"
)

Run-Exp "sc33_20k3s_sailsac_ablate_nostrictckpt" @(
    "--controller-variants","shielded_replay_warmstart_sac",
    "--train-validation-checkpoint-every","1000",
    "--train-validation-gate-dwell-threshold","0.05",
    "--train-validation-gate-violation-weight","200000",
    "--train-validation-midband-dwell-weight","5000",
    "--train-validation-soc-target-tracking-weight","2000",
    "--train-validation-peak-discharge-headroom-weight","5000",
    "--train-validation-valley-charge-weight","2000",
    "--train-validation-peak-discharge-weight","2000",
    "--train-validation-final-soc-deviation-weight","20000",
    "--train-validation-peak-reserve-weight","2500"
)

Write-Host "=== Final IEEE33 SAIL-SAC suite completed ==="
