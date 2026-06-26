$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Raw | ConvertFrom-Json
$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$r41 = @($json70.results) | Where-Object { $_.name -eq "r41_globalctx2_70" }

$cfgNames = @("stock_full_besttal", "globalctx_full_besttal", "globalctx_full_default")
$cfgLabels = @("STOCK+TAL", "ARCH+TAL", "ARCH ONLY")

function Stats($vals) {
    $mean = ($vals[0] + $vals[1] + $vals[2]) / 3
    $var = (($vals[0]-$mean)*($vals[0]-$mean) + ($vals[1]-$mean)*($vals[1]-$mean) + ($vals[2]-$mean)*($vals[2]-$mean)) / 2
    $std = [math]::Sqrt($var)
    return @{ mean = $mean; std = $std }
}

# Collect means
$means = @{}
foreach ($i in 0..2) {
    $cfg = $cfgNames[$i]
    $label = $cfgLabels[$i]
    $seeds = @($json.results) | Where-Object { $_.name -like "$($cfg)_seed*" }
    $means[$label] = @{}
    
    $metricKeys = @("mAP50_all","mAP50_95_all","precision","recall","mAP50_small","mAP50_medium","mAP50_large","AR50_small","AR50_medium","AR50_large")
    foreach ($key in $metricKeys) {
        $vals = @(); foreach ($s in $seeds) { $vals += [double]$s.metrics.$key }
        $means[$label]["m_$key"] = (Stats $vals).mean
    }
    
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        foreach ($key in @("AP50_all","AP50_95_all","AP50_small","AP50_medium","AP50_large","AP50_95_small","AP50_95_medium","AP50_95_large")) {
            $vals = @(); foreach ($s in $seeds) { $vals += [double]$s.per_class.$cls.$key }
            $means[$label]["${cls}_${key}"] = (Stats $vals).mean
        }
    }
}

$fmt = "{0,-20} {1,10} {2,14} {3,14} {4,14}"

# Overall
Write-Host "======================================================================"
Write-Host "  MEAN IMPROVEMENT vs R41 (globalctx2, 70%, default TAL)"
Write-Host "======================================================================"
Write-Host ""
Write-Host "=== OVERALL METRICS ==="
Write-Host ""
Write-Host ($fmt -f "Metric","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)

$metricPairs = @(
    @("mAP50", "mAP50_all"),
    @("mAP50-95", "mAP50_95_all"),
    @("Precision", "precision"),
    @("Recall", "recall"),
    @("mAP50_small", "mAP50_small"),
    @("mAP50_medium", "mAP50_medium"),
    @("mAP50_large", "mAP50_large")
)

foreach ($pair in $metricPairs) {
    $label = $pair[0]
    $key = $pair[1]
    $v41 = [double]$r41.metrics.$key
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["m_$key"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $label, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# Per-class AP50 overall
Write-Host ""
Write-Host "=== PER-CLASS AP50 OVERALL ==="
Write-Host ""
Write-Host ($fmt -f "Class","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_all
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["${cls}_AP50_all"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# Per-class AP50 small
Write-Host ""
Write-Host "=== PER-CLASS AP50 SMALL ==="
Write-Host ""
Write-Host ($fmt -f "Class_S","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_small
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["${cls}_AP50_small"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# Per-class AP50 medium
Write-Host ""
Write-Host "=== PER-CLASS AP50 MEDIUM ==="
Write-Host ""
Write-Host ($fmt -f "Class_M","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_medium
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["${cls}_AP50_medium"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# Per-class AP50 large
Write-Host ""
Write-Host "=== PER-CLASS AP50 LARGE ==="
Write-Host ""
Write-Host ($fmt -f "Class_L","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_large
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["${cls}_AP50_large"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# Per-class mAP50-95
Write-Host ""
Write-Host "=== PER-CLASS mAP50-95 ==="
Write-Host ""
Write-Host ($fmt -f "Class_5095","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_95_all
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["${cls}_AP50_95_all"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}

# Per-class mAP50-95 small
Write-Host ""
Write-Host "=== PER-CLASS mAP50-95 SMALL ==="
Write-Host ""
Write-Host ($fmt -f "Class_5095_S","R41","STOCK+TAL","ARCH+TAL","ARCH ONLY")
Write-Host ("-" * 80)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $v41 = [double]$r41.per_class.$cls.AP50_95_small
    $pcts = @([math]::Round($v41 * 100, 2).ToString())
    foreach ($cfgLabel in $cfgLabels) {
        $vm = $means[$cfgLabel]["${cls}_AP50_95_small"]
        $d = [math]::Round(($vm - $v41) / $v41 * 100, 2)
        $s = if ($d -gt 0) {"+"} else {""}
        $pcts += ($s + $d + "%")
    }
    Write-Host ($fmt -f $cls, $pcts[0], $pcts[1], $pcts[2], $pcts[3])
}
