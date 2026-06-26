$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Raw | ConvertFrom-Json

$cfgNames = @("stock_full_besttal", "globalctx_full_besttal", "globalctx_full_default")
$cfgLabels = @("STOCK+TAL", "ARCH+TAL", "ARCH ONLY")

function Stats($vals) {
    $mean = ($vals[0] + $vals[1] + $vals[2]) / 3
    $var = (($vals[0]-$mean)*($vals[0]-$mean) + ($vals[1]-$mean)*($vals[1]-$mean) + ($vals[2]-$mean)*($vals[2]-$mean)) / 2
    $std = [math]::Sqrt($var)
    return @{ mean = $mean; std = $std }
}

function FmtMS($s) {
    return [math]::Round($s.mean * 100, 2).ToString() + " +/- " + [math]::Round($s.std * 100, 2).ToString()
}

# Collect stats for all configs
$allStats = @{}
foreach ($i in 0..2) {
    $cfg = $cfgNames[$i]
    $label = $cfgLabels[$i]
    $seeds = @($json.results) | Where-Object { $_.name -like "$($cfg)_seed*" }
    $allStats[$label] = @{}
    
    # Overall metrics
    $metricKeys = @("mAP50_all","mAP50_95_all","precision","recall","mAP50_small","mAP50_medium","mAP50_large","AR50_small","AR50_medium","AR50_large")
    foreach ($key in $metricKeys) {
        $vals = @()
        foreach ($s in $seeds) { $vals += [double]$s.metrics.$key }
        $allStats[$label][$key] = Stats $vals
    }
    
    # Per-class
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $clsKeys = @("AP50_all","AP50_95_all","AP50_small","AP50_medium","AP50_large","AP50_95_small","AP50_95_medium","AP50_95_large")
        foreach ($key in $clsKeys) {
            $vals = @()
            foreach ($s in $seeds) { $vals += [double]$s.per_class.$cls.$key }
            $allStats[$label]["${cls}_${key}"] = Stats $vals
        }
    }
}

# Print tables
Write-Host "======================================================================"
Write-Host "  SEED VALIDATION: MEAN +/- STD (3 seeds each)"
Write-Host "======================================================================"
Write-Host ""

# Overall
Write-Host "=== OVERALL METRICS (%) ==="
Write-Host ""
$fmt = "{0,-20} {1,20} {2,20} {3,20}"
Write-Host ($fmt -f "Metric", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)

$metricLabels = @(
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

foreach ($pair in $metricLabels) {
    $label = $pair[0]
    $key = $pair[1]
    $s1 = FmtMS $allStats["STOCK+TAL"][$key]
    $s2 = FmtMS $allStats["ARCH+TAL"][$key]
    $s3 = FmtMS $allStats["ARCH ONLY"][$key]
    Write-Host ($fmt -f $label, $s1, $s2, $s3)
}

# Per-class AP50 overall
Write-Host ""
Write-Host "=== PER-CLASS AP50 OVERALL (%) ==="
Write-Host ""
Write-Host ($fmt -f "Class", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $s1 = FmtMS $allStats["STOCK+TAL"]["${cls}_AP50_all"]
    $s2 = FmtMS $allStats["ARCH+TAL"]["${cls}_AP50_all"]
    $s3 = FmtMS $allStats["ARCH ONLY"]["${cls}_AP50_all"]
    Write-Host ($fmt -f $cls, $s1, $s2, $s3)
}

# Per-class AP50 small
Write-Host ""
Write-Host "=== PER-CLASS AP50 SMALL (%) ==="
Write-Host ""
Write-Host ($fmt -f "Class", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $s1 = FmtMS $allStats["STOCK+TAL"]["${cls}_AP50_small"]
    $s2 = FmtMS $allStats["ARCH+TAL"]["${cls}_AP50_small"]
    $s3 = FmtMS $allStats["ARCH ONLY"]["${cls}_AP50_small"]
    Write-Host ($fmt -f $cls, $s1, $s2, $s3)
}

# Per-class AP50 medium
Write-Host ""
Write-Host "=== PER-CLASS AP50 MEDIUM (%) ==="
Write-Host ""
Write-Host ($fmt -f "Class", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $s1 = FmtMS $allStats["STOCK+TAL"]["${cls}_AP50_medium"]
    $s2 = FmtMS $allStats["ARCH+TAL"]["${cls}_AP50_medium"]
    $s3 = FmtMS $allStats["ARCH ONLY"]["${cls}_AP50_medium"]
    Write-Host ($fmt -f $cls, $s1, $s2, $s3)
}

# Per-class AP50 large
Write-Host ""
Write-Host "=== PER-CLASS AP50 LARGE (%) ==="
Write-Host ""
Write-Host ($fmt -f "Class", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $s1 = FmtMS $allStats["STOCK+TAL"]["${cls}_AP50_large"]
    $s2 = FmtMS $allStats["ARCH+TAL"]["${cls}_AP50_large"]
    $s3 = FmtMS $allStats["ARCH ONLY"]["${cls}_AP50_large"]
    Write-Host ($fmt -f $cls, $s1, $s2, $s3)
}

# Per-class mAP50-95
Write-Host ""
Write-Host "=== PER-CLASS mAP50-95 (%) ==="
Write-Host ""
Write-Host ($fmt -f "Class", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $s1 = FmtMS $allStats["STOCK+TAL"]["${cls}_AP50_95_all"]
    $s2 = FmtMS $allStats["ARCH+TAL"]["${cls}_AP50_95_all"]
    $s3 = FmtMS $allStats["ARCH ONLY"]["${cls}_AP50_95_all"]
    Write-Host ($fmt -f $cls, $s1, $s2, $s3)
}

# Per-class AP50-95 small
Write-Host ""
Write-Host "=== PER-CLASS mAP50-95 SMALL (%) ==="
Write-Host ""
Write-Host ($fmt -f "Class", "STOCK+TAL", "ARCH+TAL", "ARCH ONLY")
Write-Host ("-" * 85)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $s1 = FmtMS $allStats["STOCK+TAL"]["${cls}_AP50_95_small"]
    $s2 = FmtMS $allStats["ARCH+TAL"]["${cls}_AP50_95_small"]
    $s3 = FmtMS $allStats["ARCH ONLY"]["${cls}_AP50_95_small"]
    Write-Host ($fmt -f $cls, $s1, $s2, $s3)
}
