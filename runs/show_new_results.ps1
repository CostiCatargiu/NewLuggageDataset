$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$validJson = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__valid_full_dataset.json" -Raw | ConvertFrom-Json

$validMap = @{}
foreach ($r in $validJson.results) { $validMap[$r.name] = $r }

$sorted = @($json.results) | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host ("Total runs: " + $sorted.Count)
Write-Host ""
Write-Host "=== ALL RUNS RANKED BY TEST mAP50 ==="
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

# Show new runs detail
Write-Host ""
Write-Host "=== NEW RUNS DETAIL (r39, r40) ==="
Write-Host ""
$newRuns = @($json.results) | Where-Object { $_.name -match "r39|r40" } | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

foreach ($r in $newRuns) {
    $m = $r.metrics
    Write-Host (">>> " + $r.name)
    Write-Host ("    TAL: topk=" + $r.config.tal_topk + " alpha=" + $r.config.tal_alpha + " beta=" + $r.config.tal_beta)
    Write-Host ("    TEST:  mAP50=" + [math]::Round([double]$m.mAP50_all,4) + "  m5095=" + [math]::Round([double]$m.mAP50_95_all,4) + "  Prec=" + [math]::Round([double]$m.precision,4) + "  Rec=" + [math]::Round([double]$m.recall,4))
    Write-Host ("    SIZES: small=" + [math]::Round([double]$m.mAP50_small,4) + "  medium=" + [math]::Round([double]$m.mAP50_medium,4) + "  large=" + [math]::Round([double]$m.mAP50_large,4))
    Write-Host ("    AR:    AR_S=" + [math]::Round([double]$m.AR50_small,4) + "  AR_M=" + [math]::Round([double]$m.AR50_medium,4) + "  AR_L=" + [math]::Round([double]$m.AR50_large,4))
    Write-Host "    Per-class AP50 (all / small / large):"
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $c = $r.per_class.$cls
        Write-Host ("      " + $cls.PadRight(12) + [math]::Round([double]$c.AP50_all,4).ToString().PadRight(8) + " / " + [math]::Round([double]$c.AP50_small,4).ToString().PadRight(8) + " / " + [math]::Round([double]$c.AP50_large,4))
    }
    # Scoring gap
    Write-Host "    Scoring gap (AR50_S - AP50_S):"
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $c = $r.per_class.$cls
        $gap = [math]::Round([double]$c.AR50_small - [double]$c.AP50_small, 4)
        Write-Host ("      " + $cls.PadRight(12) + "gap=" + $gap)
    }
    Write-Host ""
}

# Comparison: new vs baselines
Write-Host "=== DELTAS vs KEY BASELINES ==="
Write-Host ""

$r32b = @($json.results) | Where-Object { $_.name -eq "r32b_auxdual_arch_only_70" }
$r36 = @($json.results) | Where-Object { $_.name -eq "r36_p5ctx_seed1_702" }
$gc = @($json.results) | Where-Object { $_.name -eq "r38_globalctx_703" }

$fmt2 = "{0,-48} {1,9} {2,9} {3,9} {4,9} {5,9}"
Write-Host ($fmt2 -f "Name","d_mAP50","d_m5095","d_mS","d_mL","d_othS")
Write-Host ("-" * 100)

foreach ($r in $newRuns) {
    # vs r36 (winner)
    $dm50 = [math]::Round([double]$r.metrics.mAP50_all - [double]$r36.metrics.mAP50_all, 4)
    $dm95 = [math]::Round([double]$r.metrics.mAP50_95_all - [double]$r36.metrics.mAP50_95_all, 4)
    $dms = [math]::Round([double]$r.metrics.mAP50_small - [double]$r36.metrics.mAP50_small, 4)
    $dml = [math]::Round([double]$r.metrics.mAP50_large - [double]$r36.metrics.mAP50_large, 4)
    $dos = [math]::Round([double]$r.per_class.other.AP50_small - [double]$r36.per_class.other.AP50_small, 4)
    $s50 = if($dm50 -gt 0){"+"} else {""}
    $s95 = if($dm95 -gt 0){"+"} else {""}
    $sms = if($dms -gt 0){"+"} else {""}
    $sml = if($dml -gt 0){"+"} else {""}
    $sos = if($dos -gt 0){"+"} else {""}
    Write-Host ($fmt2 -f ($r.name + " vs r36_p5ctx"), ($s50+$dm50), ($s95+$dm95), ($sms+$dms), ($sml+$dml), ($sos+$dos))
    
    # vs globalctx
    $dm50 = [math]::Round([double]$r.metrics.mAP50_all - [double]$gc.metrics.mAP50_all, 4)
    $dm95 = [math]::Round([double]$r.metrics.mAP50_95_all - [double]$gc.metrics.mAP50_95_all, 4)
    $dms = [math]::Round([double]$r.metrics.mAP50_small - [double]$gc.metrics.mAP50_small, 4)
    $dml = [math]::Round([double]$r.metrics.mAP50_large - [double]$gc.metrics.mAP50_large, 4)
    $dos = [math]::Round([double]$r.per_class.other.AP50_small - [double]$gc.per_class.other.AP50_small, 4)
    $s50 = if($dm50 -gt 0){"+"} else {""}
    $s95 = if($dm95 -gt 0){"+"} else {""}
    $sms = if($dms -gt 0){"+"} else {""}
    $sml = if($dml -gt 0){"+"} else {""}
    $sos = if($dos -gt 0){"+"} else {""}
    Write-Host ($fmt2 -f ($r.name + " vs globalctx"), ($s50+$dm50), ($s95+$dm95), ($sms+$dms), ($sml+$dml), ($sos+$dos))
    Write-Host ""
}
