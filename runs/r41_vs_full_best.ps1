$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$jsonFull = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$r41 = @($json70.results) | Where-Object { $_.name -eq "r41_globalctx2_70" }
$best = @($jsonFull.results) | Where-Object { $_.name -eq "stock_full_besttal" }

Write-Host "================================================================"
Write-Host "  r41_globalctx2 (70% ablation) vs stock_full_besttal (full)"
Write-Host "================================================================"
Write-Host ""
Write-Host "  R41:  globalctx2 arch, 70% split, default TAL (tk=10/a=0.5/b=6)"
Write-Host "  BEST: stock YOLOv12, full dataset, best TAL (tk=13/a=0.7/b=4)"
Write-Host ""

$fmt = "{0,-25} {1,12} {2,12} {3,12} {4,10}"
Write-Host ($fmt -f "Metric","R41 (70%)","Best (full)","Abs Delta","% Diff")
Write-Host ("-" * 75)

$metrics = @(
    @("mAP50", "mAP50_all"),
    @("mAP50-95", "mAP50_95_all"),
    @("Precision", "precision"),
    @("Recall", "recall"),
    @("mAP50_small", "mAP50_small"),
    @("mAP50_medium", "mAP50_medium"),
    @("mAP50_large", "mAP50_large"),
    @("AR50_small", "AR50_small"),
    @("AR50_medium", "AR50_medium"),
    @("AR50_large", "AR50_large")
)

foreach ($pair in $metrics) {
    $label = $pair[0]
    $key = $pair[1]
    $v41 = [double]$r41.metrics.$key
    $vb = [double]$best.metrics.$key
    $abs = [math]::Round($vb - $v41, 4)
    $pct = [math]::Round(($vb - $v41) / $v41 * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $label, [math]::Round($v41,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 (overall)"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class","R41 (70%)","Best (full)","Abs Delta","% Diff")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_all
    $vb = [double]$best.per_class.$cls.AP50_all
    $abs = [math]::Round($vb - $v41, 4)
    $pct = [math]::Round(($vb - $v41) / $v41 * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($v41,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 SMALL"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_small","R41 (70%)","Best (full)","Abs Delta","% Diff")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_small
    $vb = [double]$best.per_class.$cls.AP50_small
    $abs = [math]::Round($vb - $v41, 4)
    $pct = [math]::Round(($vb - $v41) / $v41 * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($v41,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 MEDIUM"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_medium","R41 (70%)","Best (full)","Abs Delta","% Diff")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_medium
    $vb = [double]$best.per_class.$cls.AP50_medium
    $abs = [math]::Round($vb - $v41, 4)
    $pct = [math]::Round(($vb - $v41) / $v41 * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($v41,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 LARGE"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_large","R41 (70%)","Best (full)","Abs Delta","% Diff")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_large
    $vb = [double]$best.per_class.$cls.AP50_large
    $abs = [math]::Round($vb - $v41, 4)
    $pct = [math]::Round(($vb - $v41) / $v41 * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($v41,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS mAP50-95"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_5095","R41 (70%)","Best (full)","Abs Delta","% Diff")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_95_all
    $vb = [double]$best.per_class.$cls.AP50_95_all
    $abs = [math]::Round($vb - $v41, 4)
    $pct = [math]::Round(($vb - $v41) / $v41 * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($v41,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}
