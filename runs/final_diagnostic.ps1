$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

# The top 3 each have a unique strength. What if we could combine them?
$r36 = @($json.results) | Where-Object { $_.name -eq "r36_p5ctx_seed1_702" }
$gc = @($json.results) | Where-Object { $_.name -eq "r38_globalctx_703" }
$r39 = @($json.results) | Where-Object { $_.name -eq "r39_combo_70" }
$r32b = @($json.results) | Where-Object { $_.name -eq "r32b_auxdual_arch_only_70" }
$base = @($json.results) | Where-Object { $_.name -eq "rev_stock_default" }

Write-Host "=== THE THREE UNIQUE STRENGTHS ==="
Write-Host ""
$fmt = "{0,-25} {1,7} {2,7} {3,7} {4,7} {5,7} {6,7}"
Write-Host ($fmt -f "Name","mAP50","m5095","mS","mL","othS","knfS")
Write-Host ("-" * 90)
foreach ($r in @($r36, $gc, $r39, $r32b, $base)) {
    Write-Host ($fmt -f $r.name, [math]::Round([double]$r.metrics.mAP50_all,4), [math]::Round([double]$r.metrics.mAP50_95_all,4), [math]::Round([double]$r.metrics.mAP50_small,4), [math]::Round([double]$r.metrics.mAP50_large,4), [math]::Round([double]$r.per_class.other.AP50_small,4), [math]::Round([double]$r.per_class.knife.AP50_small,4))
}

Write-Host ""
Write-Host "=== WHAT EACH WINNER IS BEST AT ==="
Write-Host ""
Write-Host "r36_p5ctx:   BEST mAP50 (82.65), BEST m5095 (52.93)"
Write-Host "r38_globalctx: BEST mAP50_small (66.18), BEST knife_small (67.56)"
Write-Host "r39_combo:   BEST other_small (54.14) by HUGE margin (+3.6pp)"
Write-Host ""

# What does r39 have that nobody else does? DySample.
# What does globalctx have? ZGGlobalContext per level.
# What does r36 have? p5context neck (SmallDetail@P3 + WideFuseV2@P4 + WideFuse@P5)
# r39 = globalctx + DySample. It sacrifices mAP50 for other_small.
# The sacrifice comes from DySample hurting overall precision.

Write-Host "=== R39 PRECISION/RECALL vs GLOBALCTX ==="
Write-Host ""
Write-Host ("globalctx: Prec=" + [math]::Round([double]$gc.metrics.precision,4) + "  Rec=" + [math]::Round([double]$gc.metrics.recall,4))
Write-Host ("r39_combo: Prec=" + [math]::Round([double]$r39.metrics.precision,4) + "  Rec=" + [math]::Round([double]$r39.metrics.recall,4))
Write-Host ("delta:     Prec=" + [math]::Round([double]$r39.metrics.precision - [double]$gc.metrics.precision,4) + "  Rec=" + [math]::Round([double]$r39.metrics.recall - [double]$gc.metrics.recall,4))
Write-Host ""

# Per-class: where does r39 win and lose vs globalctx?
Write-Host "=== R39 vs GLOBALCTX: PER-CLASS DETAIL ==="
Write-Host ""
$fmt2 = "{0,-12} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}"
Write-Host ($fmt2 -f "Class","GC_all","R39_all","d_all","GC_S","R39_S","d_S")
Write-Host ("-" * 70)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $ga = [math]::Round([double]$gc.per_class.$cls.AP50_all,4)
    $ra = [math]::Round([double]$r39.per_class.$cls.AP50_all,4)
    $da = [math]::Round($ra - $ga, 4)
    $gs = [math]::Round([double]$gc.per_class.$cls.AP50_small,4)
    $rs = [math]::Round([double]$r39.per_class.$cls.AP50_small,4)
    $ds = [math]::Round($rs - $gs, 4)
    $sa = if($da -gt 0){"+"} else {""}
    $ss = if($ds -gt 0){"+"} else {""}
    Write-Host ($fmt2 -f $cls, $ga, $ra, ($sa+$da), $gs, $rs, ($ss+$ds))
}

Write-Host ""
Write-Host "=== R39 vs GLOBALCTX: PER-CLASS MEDIUM + LARGE ==="
Write-Host ""
Write-Host ($fmt2 -f "Class","GC_M","R39_M","d_M","GC_L","R39_L","d_L")
Write-Host ("-" * 70)
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $gm = [math]::Round([double]$gc.per_class.$cls.AP50_medium,4)
    $rm = [math]::Round([double]$r39.per_class.$cls.AP50_medium,4)
    $dm = [math]::Round($rm - $gm, 4)
    $gl = [math]::Round([double]$gc.per_class.$cls.AP50_large,4)
    $rl = [math]::Round([double]$r39.per_class.$cls.AP50_large,4)
    $dl = [math]::Round($rl - $gl, 4)
    $sm = if($dm -gt 0){"+"} else {""}
    $sl = if($dl -gt 0){"+"} else {""}
    Write-Host ($fmt2 -f $cls, $gm, $rm, ($sm+$dm), $gl, $rl, ($sl+$dl))
}

# Now check: what if we just use DySample on ONE upsample (the P3-feeding one)
# instead of both? r39 replaced BOTH upsamples. The P5->P4 DySample might be
# hurting the P4 fusion while the P4->P3 DySample is what helps other_small.
Write-Host ""
Write-Host "=== HYPOTHESIS: SINGLE DySample (P3-feeding only) ==="
Write-Host ""
Write-Host "R39 uses DySample on BOTH FPN upsamples:"
Write-Host "  Layer 9:  DySample (P5->P4 upsample)"
Write-Host "  Layer 12: DySample (P4->P3 upsample)"
Write-Host ""
Write-Host "The P3-feeding DySample (layer 12) helps small objects"
Write-Host "  because sharper upsampling preserves detail flowing into P3."
Write-Host "The P5->P4 DySample (layer 9) might HURT medium/large objects"
Write-Host "  by oversharpening the P4 fusion."
Write-Host ""
Write-Host "Evidence: R39 vs globalctx medium-object performance:"
foreach ($cls in @("knife","long_gun","other","pistol")) {
    $gm = [math]::Round([double]$gc.per_class.$cls.AP50_medium,4)
    $rm = [math]::Round([double]$r39.per_class.$cls.AP50_medium,4)
    $dm = [math]::Round($rm - $gm, 4)
    $s = if($dm -gt 0){"+"} else {""}
    Write-Host ("  " + $cls.PadRight(12) + "medium delta = " + $s + $dm)
}
