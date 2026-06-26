$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$sorted = @($json.results) | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host ("Total runs: " + $sorted.Count)
Write-Host ""
Write-Host "=== 70% SPLIT — ALL RUNS RANKED ==="
Write-Host ""
$fmt = "{0,-3} {1,-48} {2,7} {3,7} {4,7} {5,7} {6,7} {7,6} {8,6} {9,6}"
Write-Host ($fmt -f "#","Name","mAP50","m5095","mS","mL","othS","tk","a","b")
Write-Host ("-" * 130)

$rank = 1
foreach ($r in $sorted) {
    $m = $r.metrics
    Write-Host ($fmt -f $rank, $r.name, [math]::Round([double]$m.mAP50_all,4), [math]::Round([double]$m.mAP50_95_all,4), [math]::Round([double]$m.mAP50_small,4), [math]::Round([double]$m.mAP50_large,4), [math]::Round([double]$r.per_class.other.AP50_small,4), $r.config.tal_topk, $r.config.tal_alpha, $r.config.tal_beta)
    $rank++
}

# Show new runs that weren't in previous analysis
Write-Host ""
Write-Host "=== NEW RUNS (not seen before) ==="
Write-Host ""
$known = @("rev_r21_arch_default","rev_r21_tal","rev_r21p2_default2","rev_stock_default","rev_stock_tal","r32b_auxdual_arch_only_70","r33_auxdual_p3d_arch_only_703","r34_auxdual_p3main_arch_only_70","r35_p5context_703","r35_wfv2_p3_703","r35_r34_aux075_705","r35_multiproto_702","r36_p5ctx_seed1_702","r36_p5big_702","r36_r32b_p5ctx_70","r38_bifpn_706","r38_dysample_705","r38_gather_704","r38_globalctx_703","r39_combo_70","r40_deep_p3_globalctx_703","r40_deep_p3_r32b_70")

foreach ($r in $sorted) {
    if ($r.name -notin $known) {
        $m = $r.metrics
        Write-Host (">>> " + $r.name + " (NEW)")
        Write-Host ("    TAL: topk=" + $r.config.tal_topk + " alpha=" + $r.config.tal_alpha + " beta=" + $r.config.tal_beta)
        Write-Host ("    TEST:  mAP50=" + [math]::Round([double]$m.mAP50_all,4) + "  m5095=" + [math]::Round([double]$m.mAP50_95_all,4) + "  Prec=" + [math]::Round([double]$m.precision,4) + "  Rec=" + [math]::Round([double]$m.recall,4))
        Write-Host ("    SIZES: small=" + [math]::Round([double]$m.mAP50_small,4) + "  medium=" + [math]::Round([double]$m.mAP50_medium,4) + "  large=" + [math]::Round([double]$m.mAP50_large,4))
        Write-Host "    Per-class AP50 (all / small / large):"
        foreach ($cls in @("knife","long_gun","other","pistol")) {
            $c = $r.per_class.$cls
            Write-Host ("      " + $cls.PadRight(12) + [math]::Round([double]$c.AP50_all,4).ToString().PadRight(8) + " / " + [math]::Round([double]$c.AP50_small,4).ToString().PadRight(8) + " / " + [math]::Round([double]$c.AP50_large,4))
        }
        Write-Host ""
    }
}
