$json70 = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$jsonFull = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$r41 = @($json70.results) | Where-Object { $_.name -eq "r41_globalctx2_70" }
$archOnly = @($jsonFull.results) | Where-Object { $_.name -eq "globalctx_full_default" }
$archTal = @($jsonFull.results) | Where-Object { $_.name -eq "globalctx_full_besttal" }
$talOnly = @($jsonFull.results) | Where-Object { $_.name -eq "stock_full_besttal" }

$output = @{
    description = "Master comparison: 4 configurations (R41 ablation vs full dataset variants)"
    generated = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    configurations = @{
        r41_globalctx2_70 = "globalctx2 arch, 70% split, default TAL (tk=10/a=0.5/b=6)"
        globalctx_full_default = "globalctx arch, full dataset, default TAL (tk=10/a=0.5/b=6)"
        globalctx_full_besttal = "globalctx arch, full dataset, best TAL (tk=13/a=0.7/b=4)"
        stock_full_besttal = "stock YOLOv12, full dataset, best TAL (tk=13/a=0.7/b=4)"
    }
    results = @($r41, $archOnly, $archTal, $talOnly)
}

$output | ConvertTo-Json -Depth 10 | Set-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\master_4way_comparison.json" -Encoding UTF8
Write-Host "Written to master_4way_comparison.json"
Write-Host ("File size: " + (Get-Item "C:\DISK\luggagerepo\NewLuggageDataset\runs\master_4way_comparison.json").Length + " bytes")
