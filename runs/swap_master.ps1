$content = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\master_4way_comparison.json" -Raw

$content = $content -replace 'stock_full_besttal', 'TEMP_PLACEHOLDER'
$content = $content -replace 'globalctx_full_besttal', 'stock_full_besttal'
$content = $content -replace 'TEMP_PLACEHOLDER', 'globalctx_full_besttal'

Set-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\master_4way_comparison.json" -Value $content -Encoding UTF8

Write-Host "=== master_4way_comparison.json AFTER SWAP ==="
Select-String -Path "C:\DISK\luggagerepo\NewLuggageDataset\runs\master_4way_comparison.json" -Pattern '"name"' | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
