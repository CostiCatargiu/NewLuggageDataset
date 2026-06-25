$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$all = @($json.results)
$fmt = "{0,-45} {1,7} {2,7} {3,7} {4,7} {5,7} {6,7} {7,7} {8,7}"

# 1. mAP50-95 breakdown — where is localization quality best?
Write-Host "=== mAP50-95 PER CLASS (localization quality) ==="
Write-Host ""
$sorted95 = $all | Sort-Object {[double]$_.metrics.mAP50_95_all} -Descending | Select-Object -First 8
Write-Host ("{0,-45} {1,8} {2,8} {3,8} {4,8} {5,8}" -f "Name","m5095","kn5095","gn5095","ot5095","pi5095")
Write-Host ("-" * 95)
foreach ($r in $sorted95) {
    Write-Host ("{0,-45} {1,8} {2,8} {3,8} {4,8} {5,8}" -f $r.name, [math]::Round([double]$r.metrics.mAP50_95_all,4), [math]::Round([double]$r.per_class.knife.AP50_95_all,4), [math]::Round([double]$r.per_class.long_gun.AP50_95_all,4), [math]::Round([double]$r.per_class.other.AP50_95_all,4), [math]::Round([double]$r.per_class.pistol.AP50_95_all,4))
}

# 2. mAP50 vs mAP50-95 ratio — who has worst localization relative to detection?
Write-Host ""
Write-Host "=== mAP50-95 / mAP50 RATIO (localization tightness) ==="
Write-Host ""
$sorted = $all | Sort-Object {[double]$_.metrics.mAP50_all} -Descending | Select-Object -First 10
Write-Host ("{0,-45} {1,8} {2,8} {3,8}" -f "Name","mAP50","m5095","ratio")
Write-Host ("-" * 70)
foreach ($r in $sorted) {
    $m50 = [double]$r.metrics.mAP50_all
    $m95 = [double]$r.metrics.mAP50_95_all
    $ratio = [math]::Round($m95 / $m50, 4)
    Write-Host ("{0,-45} {1,8} {2,8} {3,8}" -f $r.name, [math]::Round($m50,4), [math]::Round($m95,4), $ratio)
}

# 3. Per-class ratio breakdown for winner — which class has worst localization?
Write-Host ""
Write-Host "=== WINNER: PER-CLASS LOCALIZATION GAP (AP50 vs AP50-95) ==="
Write-Host ""
$w = $all | Where-Object { $_.name -eq "r36_p5ctx_seed1_702" }
Write-Host ("{0,-12} {1,8} {2,8} {3,8} {4,10}" -f "Class","AP50","AP5095","ratio","gap(50-5095)")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $a50 = [double]$w.per_class.$cls.AP50_all
    $a95 = [double]$w.per_class.$cls.AP50_95_all
    $ratio = [math]::Round($a95 / $a50, 4)
    $gap = [math]::Round($a50 - $a95, 4)
    Write-Host ("{0,-12} {1,8} {2,8} {3,8} {4,10}" -f $cls, [math]::Round($a50,4), [math]::Round($a95,4), $ratio, $gap)
}

# 4. Same per-class ratio but for SMALL objects
Write-Host ""
Write-Host "=== WINNER: SMALL-OBJECT LOCALIZATION GAP ==="
Write-Host ""
Write-Host ("{0,-12} {1,8} {2,8} {3,8} {4,10}" -f "Class_S","AP50_S","AP5095_S","ratio","gap")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $a50 = [double]$w.per_class.$cls.AP50_small
    $a95 = [double]$w.per_class.$cls.AP50_95_small
    $ratio = if ($a50 -gt 0) { [math]::Round($a95 / $a50, 4) } else { 0 }
    $gap = [math]::Round($a50 - $a95, 4)
    Write-Host ("{0,-12} {1,8} {2,8} {3,8} {4,10}" -f $cls, [math]::Round($a50,4), [math]::Round($a95,4), $ratio, $gap)
}

# 5. AR50 vs mAP50 — precision gap (how much recall is lost to scoring?)
Write-Host ""
Write-Host "=== RECALL vs PRECISION GAP (AR50 - mAP50 = scoring quality) ==="
Write-Host ""
$topArch = $all | Where-Object { $_.config.tal_topk -eq 10 } | Sort-Object {[double]$_.metrics.mAP50_all} -Descending | Select-Object -First 8
Write-Host ("{0,-45} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}" -f "Name","AR50_S","mAP50_S","gap_S","AR50_M","mAP50_M","gap_M")
Write-Host ("-" * 105)
foreach ($r in $topArch) {
    $m = $r.metrics
    $ars = [math]::Round([double]$m.AR50_small, 4)
    $ms = [math]::Round([double]$m.mAP50_small, 4)
    $gs = [math]::Round($ars - $ms, 4)
    $arm = [math]::Round([double]$m.AR50_medium, 4)
    $mm = [math]::Round([double]$m.mAP50_medium, 4)
    $gm = [math]::Round($arm - $mm, 4)
    Write-Host ("{0,-45} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}" -f $r.name, $ars, $ms, $gs, $arm, $mm, $gm)
}

# 6. Per-class AR50 vs AP50 for small objects — where is scoring worst?
Write-Host ""
Write-Host "=== WINNER: PER-CLASS SMALL-OBJECT SCORING GAP (AR50_S - AP50_S) ==="
Write-Host ""
Write-Host ("{0,-12} {1,8} {2,8} {3,8}" -f "Class","AR50_S","AP50_S","gap")
Write-Host ("-" * 40)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $ar = [math]::Round([double]$w.per_class.$cls.AR50_small, 4)
    $ap = [math]::Round([double]$w.per_class.$cls.AP50_small, 4)
    $g = [math]::Round($ar - $ap, 4)
    Write-Host ("{0,-12} {1,8} {2,8} {3,8}" -f $cls, $ar, $ap, $g)
}

# 7. Compare winner vs globalctx on the scoring gap
Write-Host ""
Write-Host "=== SCORING GAP: WINNER vs GLOBALCTX ==="
Write-Host ""
$gc = $all | Where-Object { $_.name -eq "r38_globalctx_703" }
Write-Host ("{0,-12} {1,10} {2,10} {3,10} {4,10}" -f "Class_S","W_gap","GC_gap","W_AP50S","GC_AP50S")
Write-Host ("-" * 55)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $war = [double]$w.per_class.$cls.AR50_small
    $wap = [double]$w.per_class.$cls.AP50_small
    $wg = [math]::Round($war - $wap, 4)
    $gar = [double]$gc.per_class.$cls.AR50_small
    $gap2 = [double]$gc.per_class.$cls.AP50_small
    $gg = [math]::Round($gar - $gap2, 4)
    Write-Host ("{0,-12} {1,10} {2,10} {3,10} {4,10}" -f $cls, $wg, $gg, [math]::Round($wap,4), [math]::Round($gap2,4))
}
