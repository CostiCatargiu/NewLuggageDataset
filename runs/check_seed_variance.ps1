# Check seed variance from 70% experiments where we have multiple seeds
$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

# r36_p5ctx ran at seed 0 (r35_p5context_703) and seed 1 (r36_p5ctx_seed1_702)
# These are the same architecture, different seeds
$s0 = @($json70.results) | Where-Object { $_.name -eq "r35_p5context_703" }
$s1 = @($json70.results) | Where-Object { $_.name -eq "r36_p5ctx_seed1_702" }

Write-Host "=== OBSERVED SEED VARIANCE (p5context seed0 vs seed1) ==="
Write-Host ""

$metrics = @(
    @("mAP50", "mAP50_all"),
    @("mAP50-95", "mAP50_95_all"),
    @("Precision", "precision"),
    @("Recall", "recall"),
    @("mAP50_small", "mAP50_small"),
    @("mAP50_medium", "mAP50_medium"),
    @("mAP50_large", "mAP50_large")
)

$fmt = "{0,-20} {1,12} {2,12} {3,12}"
Write-Host ($fmt -f "Metric","Seed0","Seed1","Delta")
Write-Host ("-" * 60)
foreach ($pair in $metrics) {
    $label = $pair[0]
    $key = $pair[1]
    $v0 = [math]::Round([double]$s0.metrics.$key, 4)
    $v1 = [math]::Round([double]$s1.metrics.$key, 4)
    $d = [math]::Round($v1 - $v0, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $label, $v0, $v1, ($s+$d))
}

Write-Host ""
Write-Host "=== PER-CLASS SEED VARIANCE ==="
Write-Host ""
Write-Host ($fmt -f "Class_AP50","Seed0","Seed1","Delta")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v0 = [math]::Round([double]$s0.per_class.$cls.AP50_all, 4)
    $v1 = [math]::Round([double]$s1.per_class.$cls.AP50_all, 4)
    $d = [math]::Round($v1 - $v0, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, $v0, $v1, ($s+$d))
}

Write-Host ""
Write-Host ($fmt -f "Class_AP50_S","Seed0","Seed1","Delta")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v0 = [math]::Round([double]$s0.per_class.$cls.AP50_small, 4)
    $v1 = [math]::Round([double]$s1.per_class.$cls.AP50_small, 4)
    $d = [math]::Round($v1 - $v0, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, $v0, $v1, ($s+$d))
}
