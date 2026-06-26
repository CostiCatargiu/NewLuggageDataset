$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Raw | ConvertFrom-Json
$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$r41 = @($json70.results) | Where-Object { $_.name -eq "r41_globalctx2_70" }

$cfgNames = @("stock_full_besttal", "globalctx_full_besttal", "globalctx_full_default")
$cfgLabels = @("TAL ONLY", "ARCH+TAL", "ARCH ONLY")

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

# Now: baseline = TAL ONLY (stock+bestTAL), show R41 and others as % vs baseline
$baseLabel = "TAL ONLY"

$fmt = "{0,-20} {1,14} {2,14} {3,14} {4,14}"

# Overall
Write-Host "======================================================================"
Write-Host "  BASELINE: TAL ONLY (stock + best TAL) = mean of 3 seeds"
Write-Host "  Showing: R41 and other configs as % vs baseline"
Write-Host "======================================================================"
Write-Host ""
Write-Host "=== OVERALL METRICS ==="
Write-Host ""
Write-Host ($fmt -f "Metric","TAL ONLY","R41 vs base","ARCH+TAL vs b","ARCH ONLY vs b")
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
    $base = $means[$baseLabel]["m_$key"]
    $baseStr = [math]::Round($base * 100, 2).ToString()
    
    $v41 = [double]$r41.metrics.$key
    $d41 = [math]::Round(($v41 - $base) / $base * 100, 2)
    $s41 = if ($d41 -gt 0) {"+"} else {""}
    
    $vat = $means["ARCH+TAL"]["m_$key"]
    $dat = [math]::Round(($vat - $base) / $base * 100, 2)
    $sat = if ($dat -gt 0) {"+"} else {""}
    
    $vao = $means["ARCH ONLY"]["m_$key"]
    $dao = [math]::Round(($vao - $base) / $base * 100, 2)
    $sao = if ($dao -gt 0) {"+"} else {""}
    
    Write-Host ($fmt -f $label, $baseStr, ($s41+$d41+"%"), ($sat+$dat+"%"), ($sao+$dao+"%"))
}

# Per-class sections
$sections = @(
    @("PER-CLASS AP50 OVERALL", "AP50_all"),
    @("PER-CLASS AP50 SMALL", "AP50_small"),
    @("PER-CLASS AP50 MEDIUM", "AP50_medium"),
    @("PER-CLASS AP50 LARGE", "AP50_large"),
    @("PER-CLASS mAP50-95", "AP50_95_all"),
    @("PER-CLASS mAP50-95 SMALL", "AP50_95_small")
)

foreach ($sec in $sections) {
    $title = $sec[0]
    $key = $sec[1]
    
    Write-Host ""
    Write-Host ("=== " + $title + " ===")
    Write-Host ""
    Write-Host ($fmt -f "Class","TAL ONLY","R41 vs base","ARCH+TAL vs b","ARCH ONLY vs b")
    Write-Host ("-" * 80)
    
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $base = $means[$baseLabel]["${cls}_${key}"]
        $baseStr = [math]::Round($base * 100, 2).ToString()
        
        $v41 = [double]$r41.per_class.$cls.$key
        $d41 = [math]::Round(($v41 - $base) / $base * 100, 2)
        $s41 = if ($d41 -gt 0) {"+"} else {""}
        
        $vat = $means["ARCH+TAL"]["${cls}_${key}"]
        $dat = [math]::Round(($vat - $base) / $base * 100, 2)
        $sat = if ($dat -gt 0) {"+"} else {""}
        
        $vao = $means["ARCH ONLY"]["${cls}_${key}"]
        $dao = [math]::Round(($vao - $base) / $base * 100, 2)
        $sao = if ($dao -gt 0) {"+"} else {""}
        
        Write-Host ($fmt -f $cls, $baseStr, ($s41+$d41+"%"), ($sat+$dat+"%"), ($sao+$dao+"%"))
    }
}
