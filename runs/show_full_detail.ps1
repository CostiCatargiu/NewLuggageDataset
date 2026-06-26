$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$validJson = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__valid_full_dataset.json" -Raw | ConvertFrom-Json

$validMap = @{}
foreach ($r in $validJson.results) { $validMap[$r.name] = $r }

$sorted = @($json.results) | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host "============================================================="
Write-Host "  FULL DATASET RESULTS (4 runs)"
Write-Host "============================================================="
Write-Host ""

foreach ($r in $sorted) {
    $m = $r.metrics
    $v = $validMap[$r.name]
    $vm = $v.metrics

    $testMean = [math]::Round(([double]$m.mAP50_all + [double]$vm.mAP50_all) / 2, 4)

    Write-Host ("--- " + $r.name + " ---")
    Write-Host ("TAL: topk=" + $r.config.tal_topk + " alpha=" + $r.config.tal_alpha + " beta=" + $r.config.tal_beta)
    Write-Host ("TEST:  mAP50=" + [math]::Round([double]$m.mAP50_all,4) + "  m5095=" + [math]::Round([double]$m.mAP50_95_all,4))
    Write-Host ("VALID: mAP50=" + [math]::Round([double]$vm.mAP50_all,4) + "  m5095=" + [math]::Round([double]$vm.mAP50_95_all,4))
    Write-Host ("MEAN:  mAP50=" + $testMean)
    Write-Host ""
}

Write-Host "============================================================="
Write-Host "  HEAD-TO-HEAD: STOCK vs GLOBALCTX (same TAL)"
Write-Host "============================================================="
Write-Host ""

$stockDef = @($json.results) | Where-Object { $_.name -eq "stock_full_default7" }
$gcDef = @($json.results) | Where-Object { $_.name -eq "globalctx_full_default" }
$stockTal = @($json.results) | Where-Object { $_.name -eq "stock_full_besttal" }
$gcTal = @($json.results) | Where-Object { $_.name -eq "globalctx_full_besttal" }

Write-Host "--- DEFAULT TAL (topk=10, a=0.5, b=6.0) ---"
Write-Host ""
$metrics = @("mAP50_all","mAP50_95_all","mAP50_small","mAP50_medium","mAP50_large","precision","recall")
$labels = @("mAP50","m5095","mS","mM","mL","Prec","Rec")
$fmt3 = "{0,-12} {1,10} {2,10} {3,10}"
Write-Host ($fmt3 -f "Metric","Stock","GlobalCtx","Delta")
Write-Host ("-" * 50)
for ($i=0; $i -lt $metrics.Count; $i++) {
    $key = $metrics[$i]
    $vs = [math]::Round([double]$stockDef.metrics.$key, 4)
    $vg = [math]::Round([double]$gcDef.metrics.$key, 4)
    $d = [math]::Round($vg - $vs, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt3 -f $labels[$i], $vs, $vg, ($s+$d))
}

Write-Host ""
Write-Host "Per-class AP50_small:"
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vs = [math]::Round([double]$stockDef.per_class.$cls.AP50_small, 4)
    $vg = [math]::Round([double]$gcDef.per_class.$cls.AP50_small, 4)
    $d = [math]::Round($vg - $vs, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt3 -f $cls, $vs, $vg, ($s+$d))
}

Write-Host ""
Write-Host "--- BEST TAL (topk=13, a=0.7, b=4.0) ---"
Write-Host ""
Write-Host ($fmt3 -f "Metric","Stock","GlobalCtx","Delta")
Write-Host ("-" * 50)
for ($i=0; $i -lt $metrics.Count; $i++) {
    $key = $metrics[$i]
    $vs = [math]::Round([double]$stockTal.metrics.$key, 4)
    $vg = [math]::Round([double]$gcTal.metrics.$key, 4)
    $d = [math]::Round($vg - $vs, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt3 -f $labels[$i], $vs, $vg, ($s+$d))
}

Write-Host ""
Write-Host "Per-class AP50_small:"
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $vs = [math]::Round([double]$stockTal.per_class.$cls.AP50_small, 4)
    $vg = [math]::Round([double]$gcTal.per_class.$cls.AP50_small, 4)
    $d = [math]::Round($vg - $vs, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt3 -f $cls, $vs, $vg, ($s+$d))
}

Write-Host ""
Write-Host "============================================================="
Write-Host "  KEY FINDING: TAL EFFECT (default vs best, same arch)"
Write-Host "============================================================="
Write-Host ""
Write-Host "--- STOCK: default vs best TAL ---"
Write-Host ""
Write-Host ($fmt3 -f "Metric","Default","BestTAL","Delta")
Write-Host ("-" * 50)
for ($i=0; $i -lt $metrics.Count; $i++) {
    $key = $metrics[$i]
    $vd = [math]::Round([double]$stockDef.metrics.$key, 4)
    $vt = [math]::Round([double]$stockTal.metrics.$key, 4)
    $d = [math]::Round($vt - $vd, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt3 -f $labels[$i], $vd, $vt, ($s+$d))
}

Write-Host ""
Write-Host "--- GLOBALCTX: default vs best TAL ---"
Write-Host ""
Write-Host ($fmt3 -f "Metric","Default","BestTAL","Delta")
Write-Host ("-" * 50)
for ($i=0; $i -lt $metrics.Count; $i++) {
    $key = $metrics[$i]
    $vd = [math]::Round([double]$gcDef.metrics.$key, 4)
    $vt = [math]::Round([double]$gcTal.metrics.$key, 4)
    $d = [math]::Round($vt - $vd, 4)
    $s = if ($d -gt 0) {"+"} else {""}
    Write-Host ($fmt3 -f $labels[$i], $vd, $vt, ($s+$d))
}
