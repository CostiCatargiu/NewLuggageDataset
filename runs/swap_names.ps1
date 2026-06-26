$content = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Raw

# Step 1: stock_full_besttal -> TEMP_PLACEHOLDER
$content = $content -replace 'stock_full_besttal', 'TEMP_PLACEHOLDER'

# Step 2: globalctx_full_besttal -> stock_full_besttal
$content = $content -replace 'globalctx_full_besttal', 'stock_full_besttal'

# Step 3: TEMP_PLACEHOLDER -> globalctx_full_besttal
$content = $content -replace 'TEMP_PLACEHOLDER', 'globalctx_full_besttal'

Set-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Value $content -Encoding UTF8

# Verify
Write-Host "=== AFTER SWAP ==="
Write-Host ""
Write-Host "Names:"
Select-String -Path "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Pattern '"name"' | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
Write-Host ""
Write-Host "Configurations:"
Select-String -Path "C:\DISK\luggagerepo\NewLuggageDataset\runs\seed_validation_3x3.json" -Pattern 'stock_full|globalctx_full' | Select-Object -First 6 | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) }
