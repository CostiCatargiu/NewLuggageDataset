$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$jsonFull = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$r41 = @($json70.results) | Where-Object { $_.name -eq "r41_globalctx2_70" }
$archTal = @($jsonFull.results) | Where-Object { $_.name -eq "globalctx_full_besttal" }
$talOnly = @($jsonFull.results) | Where-Object { $_.name -eq "stock_full_besttal" }

Write-Host "=================================================================="
Write-Host "  THREE-WAY COMPARISON"
Write-Host "=================================================================="
Write-Host ""
Write-Host "  R41:      globalctx2 arch, 70% split, default TAL"
Write-Host "  ARCH+TAL: globalctx arch, full dataset, best TAL"
Write-Host "  TAL ONLY: stock YOLOv12, full dataset, best TAL"
Write-Host ""

# Overall
Write-Host "=================================================================="
Write-Host "  OVERALL METRICS"
Write-Host "=================================================================="
Write-Host ""
$fmt = "{0,-20} {1,12} {2,12} {3,12}"
Write-Host ($fmt -f "Metric","R41","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 60)

$metrics = @(
    @("mAP50", "mAP50_all"),
    @("mAP50-95", "mAP50_95_all"),
    @("Precision", "precision"),
    @("Recall", "recall"),
    @("mAP50_small", "mAP50_small"),
    @("mAP50_medium", "mAP50_medium"),
    @("mAP50_large", "mAP50_large")
)

foreach ($pair in $metrics) {
    $label = $pair[0]
    $key = $pair[1]
    Write-Host ($fmt -f $label, [math]::Round([double]$r41.metrics.$key,4), [math]::Round([double]$archTal.metrics.$key,4), [math]::Round([double]$talOnly.metrics.$key,4))
}

# % improvement vs R41
Write-Host ""
Write-Host "=================================================================="
Write-Host "  % IMPROVEMENT vs R41"
Write-Host "=================================================================="
Write-Host ""
$fmt2 = "{0,-20} {1,15} {2,15}"
Write-Host ($fmt2 -f "Metric","ARCH+TAL vs R41","TAL ONLY vs R41")
Write-Host ("-" * 55)

foreach ($pair in $metrics) {
    $label = $pair[0]
    $key = $pair[1]
    $v41 = [double]$r41.metrics.$key
    $vat = [double]$archTal.metrics.$key
    $vto = [double]$talOnly.metrics.$key
    $pat = [math]::Round(($vat - $v41) / $v41 * 100, 2)
    $pto = [math]::Round(($vto - $v41) / $v41 * 100, 2)
    $sat = if ($pat -gt 0) {"+"} else {""}
    $sto = if ($pto -gt 0) {"+"} else {""}
    Write-Host ($fmt2 -f $label, ($sat+$pat+"%"), ($sto+$pto+"%"))
}

# Per-class AP50 overall
Write-Host ""
Write-Host "=================================================================="
Write-Host "  PER-CLASS AP50 (overall)"
Write-Host "=================================================================="
Write-Host ""
Write-Host ($fmt -f "Class","R41","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    Write-Host ($fmt -f $cls, [math]::Round([double]$r41.per_class.$cls.AP50_all,4), [math]::Round([double]$archTal.per_class.$cls.AP50_all,4), [math]::Round([double]$talOnly.per_class.$cls.AP50_all,4))
}

Write-Host ""
Write-Host ($fmt2 -f "Class","ARCH+TAL vs R41","TAL ONLY vs R41")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_all
    $vat = [double]$archTal.per_class.$cls.AP50_all
    $vto = [double]$talOnly.per_class.$cls.AP50_all
    $pat = [math]::Round(($vat - $v41) / $v41 * 100, 2)
    $pto = [math]::Round(($vto - $v41) / $v41 * 100, 2)
    $sat = if ($pat -gt 0) {"+"} else {""}
    $sto = if ($pto -gt 0) {"+"} else {""}
    Write-Host ($fmt2 -f $cls, ($sat+$pat+"%"), ($sto+$pto+"%"))
}

# Per-class AP50 small
Write-Host ""
Write-Host "=================================================================="
Write-Host "  PER-CLASS AP50 SMALL"
Write-Host "=================================================================="
Write-Host ""
Write-Host ($fmt -f "Class_S","R41","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    Write-Host ($fmt -f $cls, [math]::Round([double]$r41.per_class.$cls.AP50_small,4), [math]::Round([double]$archTal.per_class.$cls.AP50_small,4), [math]::Round([double]$talOnly.per_class.$cls.AP50_small,4))
}

Write-Host ""
Write-Host ($fmt2 -f "Class_S","ARCH+TAL vs R41","TAL ONLY vs R41")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_small
    $vat = [double]$archTal.per_class.$cls.AP50_small
    $vto = [double]$talOnly.per_class.$cls.AP50_small
    $pat = [math]::Round(($vat - $v41) / $v41 * 100, 2)
    $pto = [math]::Round(($vto - $v41) / $v41 * 100, 2)
    $sat = if ($pat -gt 0) {"+"} else {""}
    $sto = if ($pto -gt 0) {"+"} else {""}
    Write-Host ($fmt2 -f $cls, ($sat+$pat+"%"), ($sto+$pto+"%"))
}

# Per-class AP50 medium
Write-Host ""
Write-Host "=================================================================="
Write-Host "  PER-CLASS AP50 MEDIUM"
Write-Host "=================================================================="
Write-Host ""
Write-Host ($fmt -f "Class_M","R41","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    Write-Host ($fmt -f $cls, [math]::Round([double]$r41.per_class.$cls.AP50_medium,4), [math]::Round([double]$archTal.per_class.$cls.AP50_medium,4), [math]::Round([double]$talOnly.per_class.$cls.AP50_medium,4))
}

Write-Host ""
Write-Host ($fmt2 -f "Class_M","ARCH+TAL vs R41","TAL ONLY vs R41")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_medium
    $vat = [double]$archTal.per_class.$cls.AP50_medium
    $vto = [double]$talOnly.per_class.$cls.AP50_medium
    $pat = [math]::Round(($vat - $v41) / $v41 * 100, 2)
    $pto = [math]::Round(($vto - $v41) / $v41 * 100, 2)
    $sat = if ($pat -gt 0) {"+"} else {""}
    $sto = if ($pto -gt 0) {"+"} else {""}
    Write-Host ($fmt2 -f $cls, ($sat+$pat+"%"), ($sto+$pto+"%"))
}

# Per-class AP50 large
Write-Host ""
Write-Host "=================================================================="
Write-Host "  PER-CLASS AP50 LARGE"
Write-Host "=================================================================="
Write-Host ""
Write-Host ($fmt -f "Class_L","R41","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    Write-Host ($fmt -f $cls, [math]::Round([double]$r41.per_class.$cls.AP50_large,4), [math]::Round([double]$archTal.per_class.$cls.AP50_large,4), [math]::Round([double]$talOnly.per_class.$cls.AP50_large,4))
}

Write-Host ""
Write-Host ($fmt2 -f "Class_L","ARCH+TAL vs R41","TAL ONLY vs R41")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_large
    $vat = [double]$archTal.per_class.$cls.AP50_large
    $vto = [double]$talOnly.per_class.$cls.AP50_large
    $pat = [math]::Round(($vat - $v41) / $v41 * 100, 2)
    $pto = [math]::Round(($vto - $v41) / $v41 * 100, 2)
    $sat = if ($pat -gt 0) {"+"} else {""}
    $sto = if ($pto -gt 0) {"+"} else {""}
    Write-Host ($fmt2 -f $cls, ($sat+$pat+"%"), ($sto+$pto+"%"))
}

# mAP50-95
Write-Host ""
Write-Host "=================================================================="
Write-Host "  PER-CLASS mAP50-95"
Write-Host "=================================================================="
Write-Host ""
Write-Host ($fmt -f "Class_5095","R41","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 60)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    Write-Host ($fmt -f $cls, [math]::Round([double]$r41.per_class.$cls.AP50_95_all,4), [math]::Round([double]$archTal.per_class.$cls.AP50_95_all,4), [math]::Round([double]$talOnly.per_class.$cls.AP50_95_all,4))
}

Write-Host ""
Write-Host ($fmt2 -f "Class_5095","ARCH+TAL vs R41","TAL ONLY vs R41")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_95_all
    $vat = [double]$archTal.per_class.$cls.AP50_95_all
    $vto = [double]$talOnly.per_class.$cls.AP50_95_all
    $pat = [math]::Round(($vat - $v41) / $v41 * 100, 2)
    $pto = [math]::Round(($vto - $v41) / $v41 * 100, 2)
    $sat = if ($pat -gt 0) {"+"} else {""}
    $sto = if ($pto -gt 0) {"+"} else {""}
    Write-Host ($fmt2 -f $cls, ($sat+$pat+"%"), ($sto+$pto+"%"))
}
