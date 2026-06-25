$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$archRuns = @($json.results) | Where-Object { $_.config.tal_topk -eq 10 -and $_.config.tal_alpha -eq 0.5 -and $_.config.tal_beta -eq 6.0 } | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host "=== PURE ARCH RUNS (default TAL only) ==="
Write-Host ""
$fmt1 = "{0,-45} {1,7} {2,7} {3,7} {4,7} {5,7}"
Write-Host ($fmt1 -f "Name","mAP50","m5095","mS","mL","othS")
Write-Host ("-" * 100)
foreach ($r in $archRuns) {
    $m = $r.metrics
    Write-Host ($fmt1 -f $r.name, [math]::Round([double]$m.mAP50_all,4), [math]::Round([double]$m.mAP50_95_all,4), [math]::Round([double]$m.mAP50_small,4), [math]::Round([double]$m.mAP50_large,4), [math]::Round([double]$r.per_class.other.AP50_small,4))
}

Write-Host ""
Write-Host "=== WINNER vs BASELINE per-class per-size ==="
Write-Host ""

$winner = $archRuns[0]
$base = @($json.results) | Where-Object { $_.name -eq "rev_stock_default" }

$fmt2 = "{0,-12} {1,10} {2,10} {3,10}"
foreach ($sz in @("all","small","medium","large")) {
    $key = if ($sz -eq "all") { "AP50_all" } else { "AP50_$sz" }
    Write-Host "--- AP50_$sz ---"
    Write-Host ($fmt2 -f "Class","Baseline","Winner","Delta")
    Write-Host ("-" * 50)
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $vb = [math]::Round([double]$base.per_class.$cls.$key, 4)
        $vw = [math]::Round([double]$winner.per_class.$cls.$key, 4)
        $d = [math]::Round($vw - $vb, 4)
        $s = if ($d -gt 0) {"+"} else {""}
        Write-Host ($fmt2 -f $cls, $vb, $vw, ($s+$d))
    }
    Write-Host ""
}

Write-Host "=== GLOBALCTX vs WINNER ==="
Write-Host ""
$gc = @($json.results) | Where-Object { $_.name -eq "r38_globalctx_703" }
foreach ($sz in @("all","small")) {
    $key = if ($sz -eq "all") { "AP50_all" } else { "AP50_$sz" }
    Write-Host "--- AP50_$sz ---"
    Write-Host ($fmt2 -f "Class","Winner","GlobalCtx","Delta")
    Write-Host ("-" * 50)
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $vw = [math]::Round([double]$winner.per_class.$cls.$key, 4)
        $vg = [math]::Round([double]$gc.per_class.$cls.$key, 4)
        $d = [math]::Round($vg - $vw, 4)
        $s = if ($d -gt 0) {"+"} else {""}
        Write-Host ($fmt2 -f $cls, $vw, $vg, ($s+$d))
    }
    Write-Host ""
}

Write-Host "=== RECALL (AR50) ==="
Write-Host ""
$fmt3 = "{0,-45} {1,8} {2,8} {3,8}"
Write-Host ($fmt3 -f "Name","AR_S","AR_M","AR_L")
Write-Host ("-" * 75)
foreach ($r in $archRuns | Select-Object -First 8) {
    $m = $r.metrics
    Write-Host ($fmt3 -f $r.name, [math]::Round([double]$m.AR50_small,4), [math]::Round([double]$m.AR50_medium,4), [math]::Round([double]$m.AR50_large,4))
}

Write-Host ""
Write-Host "=== PRECISION vs RECALL ==="
Write-Host ""
$fmt4 = "{0,-45} {1,8} {2,8}"
Write-Host ($fmt4 -f "Name","Prec","Recall")
Write-Host ("-" * 65)
foreach ($r in $archRuns | Select-Object -First 8) {
    $m = $r.metrics
    Write-Host ($fmt4 -f $r.name, [math]::Round([double]$m.precision,4), [math]::Round([double]$m.recall,4))
}
