$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$jsonFull = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$r41 = @($json70.results) | Where-Object { $_.name -eq "r41_globalctx2_70" }
$archOnly = @($jsonFull.results) | Where-Object { $_.name -eq "globalctx_full_default" }
$archTal = @($jsonFull.results) | Where-Object { $_.name -eq "globalctx_full_besttal" }
$talOnly = @($jsonFull.results) | Where-Object { $_.name -eq "stock_full_besttal" }

$all = @(
    @{ label = "R41 (70%)"; data = $r41 },
    @{ label = "ARCH ONLY"; data = $archOnly },
    @{ label = "ARCH+TAL"; data = $archTal },
    @{ label = "TAL ONLY"; data = $talOnly }
)

Write-Host "======================================================================================================"
Write-Host "  MASTER TABLE: R41 vs ARCH ONLY vs ARCH+TAL vs TAL ONLY"
Write-Host "======================================================================================================"
Write-Host ""
Write-Host "  R41:       globalctx2 arch, 70% split, default TAL (tk=10/a=0.5/b=6)"
Write-Host "  ARCH ONLY: globalctx arch, full dataset, default TAL (tk=10/a=0.5/b=6)"
Write-Host "  ARCH+TAL:  globalctx arch, full dataset, best TAL (tk=13/a=0.7/b=4)"
Write-Host "  TAL ONLY:  stock YOLOv12, full dataset, best TAL (tk=13/a=0.7/b=4)"
Write-Host ""

$fmt = "{0,-20} {1,12} {2,12} {3,12} {4,12}"
$fmtP = "{0,-20} {1,12} {2,12} {3,12} {4,12}"

# OVERALL
Write-Host "======================================================================================================"
Write-Host "  OVERALL METRICS"
Write-Host "======================================================================================================"
Write-Host ""
Write-Host ($fmt -f "Metric","R41 (70%)","ARCH ONLY","ARCH+TAL","TAL ONLY")
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
    $vals = @()
    foreach ($item in $all) {
        $vals += [math]::Round([double]$item.data.metrics.$key, 4)
    }
    Write-Host ($fmt -f $label, $vals[0], $vals[1], $vals[2], $vals[3])
}

# % vs R41
Write-Host ""
Write-Host "  % IMPROVEMENT vs R41:"
Write-Host ""
Write-Host ($fmt -f "Metric","R41 (base)","ARCH ONLY","ARCH+TAL","TAL ONLY")
Write-Host ("-" * 75)
foreach ($pair in $metrics) {
    $label = $pair[0]
    $key = $pair[1]
    $v41 = [double]$r41.metrics.$key
    $pcts = @("---")
    foreach ($item in @($archOnly, $archTal, $talOnly)) {
        $v = [double]$item.metrics.$key
        $p = [math]::Round(($v - $v41) / $v41 * 100, 2)
        $s = if ($p -gt 0) {"+"} else {""}
        $pcts += ($s + $p + "%")
    }
    Write-Host ($fmt -f $label, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# PER-CLASS tables
$sizes = @(
    @("AP50 (overall)", "AP50_all"),
    @("AP50 SMALL", "AP50_small"),
    @("AP50 MEDIUM", "AP50_medium"),
    @("AP50 LARGE", "AP50_large"),
    @("mAP50-95", "AP50_95_all")
)

foreach ($sz in $sizes) {
    $title = $sz[0]
    $key = $sz[1]

    Write-Host ""
    Write-Host "======================================================================================================"
    Write-Host ("  PER-CLASS " + $title)
    Write-Host "======================================================================================================"
    Write-Host ""
    Write-Host ($fmt -f "Class","R41 (70%)","ARCH ONLY","ARCH+TAL","TAL ONLY")
    Write-Host ("-" * 75)
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $vals = @()
        foreach ($item in $all) {
            $vals += [math]::Round([double]$item.data.per_class.$cls.$key, 4)
        }
        Write-Host ($fmt -f $cls, $vals[0], $vals[1], $vals[2], $vals[3])
    }

    Write-Host ""
    Write-Host "  % vs R41:"
    Write-Host ""
    Write-Host ($fmt -f "Class","R41 (base)","ARCH ONLY","ARCH+TAL","TAL ONLY")
    Write-Host ("-" * 75)
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $v41 = [double]$r41.per_class.$cls.$key
        $pcts = @("---")
        foreach ($item in @($archOnly, $archTal, $talOnly)) {
            $v = [double]$item.per_class.$cls.$key
            $p = if ($v41 -ne 0) { [math]::Round(($v - $v41) / $v41 * 100, 2) } else { 0 }
            $s = if ($p -gt 0) {"+"} else {""}
            $pcts += ($s + $p + "%")
        }
        Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
    }
}
