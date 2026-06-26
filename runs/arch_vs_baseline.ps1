$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$base = @($json.results) | Where-Object { $_.name -eq "stock_full_default7" }
$arch = @($json.results) | Where-Object { $_.name -eq "globalctx_full_default" }

Write-Host "================================================================"
Write-Host "  IMPROVEMENT: globalctx + default TAL vs stock + default TAL"
Write-Host "  (pure architecture effect, same TAL)"
Write-Host ""
Write-Host "  BASELINE: stock_full_default (stock YOLOv12, default TAL)"
Write-Host "  ARCH:     globalctx_full_default (globalctx arch, default TAL)"
Write-Host "================================================================"
Write-Host ""

$fmt = "{0,-25} {1,12} {2,12} {3,12} {4,10}"
Write-Host ($fmt -f "Metric","Baseline","Arch","Abs Delta","% Improv")
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
    $vd = [double]$base.metrics.$key
    $vb = [double]$arch.metrics.$key
    $abs = [math]::Round($vb - $vd, 4)
    $pct = [math]::Round(($vb - $vd) / $vd * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $label, [math]::Round($vd,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 (overall)"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class","Baseline","Arch","Abs Delta","% Improv")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vd = [double]$base.per_class.$cls.AP50_all
    $vb = [double]$arch.per_class.$cls.AP50_all
    $abs = [math]::Round($vb - $vd, 4)
    $pct = [math]::Round(($vb - $vd) / $vd * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($vd,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 SMALL"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_small","Baseline","Arch","Abs Delta","% Improv")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vd = [double]$base.per_class.$cls.AP50_small
    $vb = [double]$arch.per_class.$cls.AP50_small
    $abs = [math]::Round($vb - $vd, 4)
    $pct = [math]::Round(($vb - $vd) / $vd * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($vd,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 MEDIUM"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_medium","Baseline","Arch","Abs Delta","% Improv")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vd = [double]$base.per_class.$cls.AP50_medium
    $vb = [double]$arch.per_class.$cls.AP50_medium
    $abs = [math]::Round($vb - $vd, 4)
    $pct = [math]::Round(($vb - $vd) / $vd * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($vd,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS AP50 LARGE"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_large","Baseline","Arch","Abs Delta","% Improv")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vd = [double]$base.per_class.$cls.AP50_large
    $vb = [double]$arch.per_class.$cls.AP50_large
    $abs = [math]::Round($vb - $vd, 4)
    $pct = [math]::Round(($vb - $vd) / $vd * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($vd,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}

Write-Host ""
Write-Host "================================================================"
Write-Host "  PER-CLASS mAP50-95"
Write-Host "================================================================"
Write-Host ""
Write-Host ($fmt -f "Class_5095","Baseline","Arch","Abs Delta","% Improv")
Write-Host ("-" * 75)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vd = [double]$base.per_class.$cls.AP50_95_all
    $vb = [double]$arch.per_class.$cls.AP50_95_all
    $abs = [math]::Round($vb - $vd, 4)
    $pct = [math]::Round(($vb - $vd) / $vd * 100, 2)
    $sa = if ($abs -gt 0) {"+"} else {""}
    $sp = if ($pct -gt 0) {"+"} else {""}
    Write-Host ($fmt -f $cls, [math]::Round($vd,4), [math]::Round($vb,4), ($sa+$abs), ($sp+$pct+"%"))
}
