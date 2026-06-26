$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Raw | ConvertFrom-Json

$configs = @("stock_full_besttal", "globalctx_full_besttal", "globalctx_full_default")

Write-Host "=== SEED VALIDATION: MEAN +/- STD ==="
Write-Host ""

foreach ($cfg in $configs) {
    $seeds = @($json.results) | Where-Object { $_.name -like "$($cfg)_seed*" }
    
    Write-Host ("--- " + $cfg + " (3 seeds) ---")
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
    
    $fmt = "{0,-20} {1,8} {2,8} {3,8} {4,12} {5,8}"
    Write-Host ($fmt -f "Metric","Seed0","Seed1","Seed2","Mean +/- Std","Range")
    Write-Host ("-" * 75)
    
    foreach ($pair in $metrics) {
        $label = $pair[0]
        $key = $pair[1]
        $vals = @()
        foreach ($s in $seeds) { $vals += [double]$s.metrics.$key }
        $mean = ($vals[0] + $vals[1] + $vals[2]) / 3
        $variance = (($vals[0]-$mean)*($vals[0]-$mean) + ($vals[1]-$mean)*($vals[1]-$mean) + ($vals[2]-$mean)*($vals[2]-$mean)) / 3
        $std = [math]::Sqrt($variance)
        $range = [math]::Round(($vals | Measure-Object -Maximum).Maximum - ($vals | Measure-Object -Minimum).Minimum, 4)
        $meanStr = [math]::Round($mean, 4).ToString() + " +/- " + [math]::Round($std, 4).ToString()
        Write-Host ($fmt -f $label, [math]::Round($vals[0],4), [math]::Round($vals[1],4), [math]::Round($vals[2],4), $meanStr, $range)
    }
    
    Write-Host ""
    Write-Host "  Per-class AP50_small:"
    $fmt2 = "{0,-20} {1,8} {2,8} {3,8} {4,12} {5,8}"
    Write-Host ($fmt2 -f "Class","Seed0","Seed1","Seed2","Mean +/- Std","Range")
    Write-Host ("-" * 75)
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $vals = @()
        foreach ($s in $seeds) { $vals += [double]$s.per_class.$cls.AP50_small }
        $mean = ($vals[0] + $vals[1] + $vals[2]) / 3
        $variance = (($vals[0]-$mean)*($vals[0]-$mean) + ($vals[1]-$mean)*($vals[1]-$mean) + ($vals[2]-$mean)*($vals[2]-$mean)) / 3
        $std = [math]::Sqrt($variance)
        $range = [math]::Round(($vals | Measure-Object -Maximum).Maximum - ($vals | Measure-Object -Minimum).Minimum, 4)
        $meanStr = [math]::Round($mean, 4).ToString() + " +/- " + [math]::Round($std, 4).ToString()
        Write-Host ($fmt2 -f $cls, [math]::Round($vals[0],4), [math]::Round($vals[1],4), [math]::Round($vals[2],4), $meanStr, $range)
    }
    Write-Host ""
}
