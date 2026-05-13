# Autonomous Monitor for Microgrid DRL Suite
# This script polls for experiment completion and automatically runs enrichment/compilation.

param(
    [string]$ResultsRoot = "results",
    [string]$OracleRef = "results/oracle_reference_suite_ieee33_simple_2024/oracle_reference_windows.csv",
    [int]$IntervalSeconds = 300,
    [datetime]$SuiteStartTime = [datetime]"2026-05-07 16:27:00"
)

$targetExps = @(
    "sc33_20k3s_plain_maincmp",
    "sc33_20k3s_shielded_maincmp",
    "sc33_20k3s_sailsac_maincmp",
    "sc33_20k3s_sailsac_ablate_nosafebc",
    "sc33_20k3s_sailsac_ablate_noreplaywarm",
    "sc33_20k3s_sailsac_ablate_nostrictckpt"
)

Write-Host "[Monitor] Starting microgrid suite monitor..."
Write-Host "[Monitor] Interval: $IntervalSeconds seconds"
Write-Host "[Monitor] Suite Start Time: $SuiteStartTime"
Write-Host "[Monitor] Target Experiments: $($targetExps -join ', ')"

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    foreach ($exp in $targetExps) {
        $expDir = Join-Path $ResultsRoot $exp
        $summaryPath = Join-Path $expDir "summary.csv"
        $markerPath = Join-Path $expDir ".enriched_done_new"
        
        if (Test-Path $summaryPath) {
            $summaryFile = Get-Item $summaryPath
            if ($summaryFile.LastWriteTime -ge $SuiteStartTime) {
                if (!(Test-Path $markerPath)) {
                    Write-Host "[$timestamp] Found NEW summary for $($exp) (Modified: $($summaryFile.LastWriteTime)). Enriching..."
                    
                    try {
                        # 1. Run Enrichment (Generates reviewer_*.csv)
                        python scripts/analysis/enrich_safe_warmstart_summary.py `
                            $expDir `
                            --oracle-reference-csv $OracleRef `
                            --overwrite-existing `
                            --groupby-columns "case,regime,controller_variant,train_model,test_model,eval_window_label"
                        
                        # 2. Create completion marker
                        New-Item -ItemType File -Path $markerPath -Force | Out-Null
                        
                        Write-Host "[$timestamp] Successfully enriched $($exp)."
                        
                        # 3. Trigger paper figure updates if needed
                        if ($exp -eq "sc33_20k3s_sailsac_maincmp") {
                             Write-Host "[$timestamp] Main SAIL-SAC done. Running paper figures..."
                             python scripts/plot/paper_case_study_figures.py
                        }
                    } catch {
                        Write-Host "[$timestamp] Error processing $($exp): $_"
                    }
                }
            }
        }
    }
    
    Start-Sleep -Seconds $IntervalSeconds
}
