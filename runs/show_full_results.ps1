$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$validJson = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__valid_full_dataset.json" -Raw | ConvertFrom-Json

$validMap = @{}
foreach ($r in $validJson.results) { $validMap[$r.name] = $r }

$sorted = @($json.results) | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host ("Total runs: " + $sorted.Count)
Write-Host ""
Write-Host "=== FULL DATASET RESULTS — RANKED BY TEST mAP50 ==="
Write-Host ""
$fmt = "{0,-3} {1,-48} {2,7} {3,7} {4,7} {5,7} {6,7} {7,7}"
Write-Host ($fmt -f "#","Name","mAP50","m5095","mS","mL","othS","V_mAP50")
Write-Host ("-" * 115)

$rank = 1
foreach ($r in $sorted) {
    $m = $r.metrics
    $vm = if ($validMap[$r.name]) { [math]::Round([double]$validMap[$r.name].metrics.mAP50_all,4) } else { "N/A" }
    Write-Host ($fmt -f $rank, $r.name, [math]::Round([double]$m.mAP50_all,4), [math]::Round([double]$m.mAP50_95_all,4), [math]::Round([double]$m.mAP50_small,4), [math]::Round([double]$m.mAP50_large,4), [math]::Round([double]$r.per_class.other.AP50_small,4), $vm)
    $rank++
}

Write-Host ""
Write-Host "=== TOP 5 DETAIL ==="
Write-Host ""

$top5 = $sorted | Select-Object -First 5
foreach ($r in $top5) {
    $m = $r.metrics
    Write-Host (">>> " + $r.name)
    Write-Host ("    TAL: topk=" + $r.config.tal_topk + " alpha=" + $r.config.tal_alpha + " beta=" + $r.config.tal_beta)
    Write-Host ("    TEST:  mAP50=" + [math]::Round([double]$m.mAP50_all,4) + "  m5095=" + [math]::Round([double]$m.mAP50_95_all,4) + "  Prec=" + [math]::Round([double]$m.precision,4) + "  Rec=" + [math]::Round([double]$m.recall,4))
    Write-Host ("    SIZES: small=" + [math]::Round([double]$m.mAP50_small,4) + "  medium=" + [math]::Round([double]$m.mAP50_medium,4) + "  large=" + [math]::Round([double]$m.mAP50_large,4))
    if ($validMap[$r.name]) {
        $vm = $validMap[$r.name].metrics
        Write-Host ("    VALID: mAP50=" + [math]::Round([double]$vm.mAP50_all,4) + "  m5095=" + [math]::Round([double]$vm.mAP50_95_all,4) + "  small=" + [math]::Round([double]$vm.mAP50_small,4))
    }
    Write-Host "    Per-class AP50 (all / small / large):"
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $c = $r.per_class.$cls
        Write-Host ("      " + $cls.PadRight(12) + [math]::Round([double]$c.AP50_all,4).ToString().PadRight(8) + " / " + [math]::Round([double]$c.AP50_small,4).ToString().PadRight(8) + " / " + [math]::Round([double]$c.AP50_large,4))
    }
    Write-Host ""
}
