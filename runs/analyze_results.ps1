$testRaw = [System.IO.File]::ReadAllText("C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json")
$testData = $testRaw | ConvertFrom-Json
$validRaw = [System.IO.File]::ReadAllText("C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__valid_full_dataset.json")
$validData = $validRaw | ConvertFrom-Json

# Build valid lookup
$validMap = @{}
foreach ($r in $validData.results) {
    $validMap[$r.name] = $r
}

$sorted = @($testData.results) | Sort-Object { [double]$_.metrics.mAP50_all } -Descending

Write-Host "=============================================================================================="
Write-Host "ALL 19 RUNS -- RANKED BY TEST mAP50 (with validation cross-check)"
Write-Host "=============================================================================================="
Write-Host ""
Write-Host ("{0,-3} {1,-45} {2,8} {3,8} {4,8} {5,8} {6,8} | {7,8} {8,8}" -f "#", "Name", "T_mAP50", "T_m5095", "T_mS", "T_mL", "T_othS", "V_mAP50", "V_mS")
Write-Host ("-" * 120)

$rank = 1
foreach ($r in $sorted) {
    $m = $r.metrics
    $v = $validMap[$r.name]
    $vm50 = if ($v) { [math]::Round([double]$v.metrics.mAP50_all, 4) } else { "N/A" }
    $vms = if ($v) { [math]::Round([double]$v.metrics.mAP50_small, 4) } else { "N/A" }
    
    Write-Host ("{0,-3} {1,-45} {2,8} {3,8} {4,8} {5,8} {6,8} | {7,8} {8,8}" -f $rank, $r.name, [math]::Round([double]$m.mAP50_all, 4), [math]::Round([double]$m.mAP50_95_all, 4), [math]::Round([double]$m.mAP50_small, 4), [math]::Round([double]$m.mAP50_large, 4), [math]::Round([double]$r.per_class.other.AP50_small, 4), $vm50, $vms)
    $rank++
}

Write-Host ""
Write-Host "=============================================================================================="
Write-Host "DETAILED TOP-5 BREAKDOWN"
Write-Host "=============================================================================================="
Write-Host ""

$top5 = $sorted | Select-Object -First 5
foreach ($r in $top5) {
    $m = $r.metrics
    $v = $validMap[$r.name]
    Write-Host (">>> {0}" -f $r.name)
    Write-Host ("    TAL: topk={0} alpha={1} beta={2}" -f $r.config.tal_topk, $r.config.tal_alpha, $r.config.tal_beta)
    Write-Host ("    TEST:  mAP50={0}  m5095={1}  Prec={2}  Rec={3}" -f [math]::Round([double]$m.mAP50_all,4), [math]::Round([double]$m.mAP50_95_all,4), [math]::Round([double]$m.precision,4), [math]::Round([double]$m.recall,4))
    Write-Host ("    SIZES: small={0}  medium={1}  large={2}" -f [math]::Round([double]$m.mAP50_small,4), [math]::Round([double]$m.mAP50_medium,4), [math]::Round([double]$m.mAP50_large,4))
    if ($v) {
        $vm = $v.metrics
        Write-Host ("    VALID: mAP50={0}  m5095={1}  small={2}  large={3}" -f [math]::Round([double]$vm.mAP50_all,4), [math]::Round([double]$vm.mAP50_95_all,4), [math]::Round([double]$vm.mAP50_small,4), [math]::Round([double]$vm.mAP50_large,4))
    }
    Write-Host "    Per-class AP50 (all / small / large):"
    foreach ($cls in @("knife","long_gun","other","pistol")) {
        $c = $r.per_class.$cls
        Write-Host ("      {0,-12} {1,7} / {2,7} / {3,7}" -f $cls, [math]::Round([double]$c.AP50_all,4), [math]::Round([double]$c.AP50_small,4), [math]::Round([double]$c.AP50_large,4))
    }
    Write-Host ""
}

Write-Host "=============================================================================================="
Write-Host "NEW ARCHITECTURES (R36, R38)"
Write-Host "=============================================================================================="
Write-Host ""

$newRuns = @($testData.results) | Where-Object { $_.name -match "r36|r38" } | Sort-Object { [double]$_.metrics.mAP50_all } -Descending

Write-Host ("{0,-40} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}" -f "Name", "mAP50", "m5095", "mS", "mL", "othAll", "othS")
Write-Host ("-" * 100)
foreach ($r in $newRuns) {
    $m = $r.metrics
    Write-Host ("{0,-40} {1,8} {2,8} {3,8} {4,8} {5,8} {6,8}" -f $r.name, [math]::Round([double]$m.mAP50_all,4), [math]::Round([double]$m.mAP50_95_all,4), [math]::Round([double]$m.mAP50_small,4), [math]::Round([double]$m.mAP50_large,4), [math]::Round([double]$r.per_class.other.AP50_all,4), [math]::Round([double]$r.per_class.other.AP50_small,4))
}

Write-Host ""
Write-Host "=============================================================================================="
Write-Host "DELTA: NEW ARCHS vs R32B (previous best = 82.58 mAP50)"
Write-Host "=============================================================================================="
Write-Host ""

$r32b = @($testData.results) | Where-Object { $_.name -eq "r32b_auxdual_arch_only_70" }

Write-Host ("{0,-40} {1,9} {2,9} {3,9} {4,9} {5,9}" -f "Name", "d_mAP50", "d_m5095", "d_mS", "d_mL", "d_othS")
Write-Host ("-" * 100)
foreach ($r in $newRuns) {
    $dm50 = [math]::Round([double]$r.metrics.mAP50_all - [double]$r32b.metrics.mAP50_all, 4)
    $dm95 = [math]::Round([double]$r.metrics.mAP50_95_all - [double]$r32b.metrics.mAP50_95_all, 4)
    $dms = [math]::Round([double]$r.metrics.mAP50_small - [double]$r32b.metrics.mAP50_small, 4)
    $dml = [math]::Round([double]$r.metrics.mAP50_large - [double]$r32b.metrics.mAP50_large, 4)
    $dos = [math]::Round([double]$r.per_class.other.AP50_small - [double]$r32b.per_class.other.AP50_small, 4)
    $s50 = if ($dm50 -gt 0) { "+" } else { "" }
    $s95 = if ($dm95 -gt 0) { "+" } else { "" }
    $sms = if ($dms -gt 0) { "+" } else { "" }
    $sml = if ($dml -gt 0) { "+" } else { "" }
    $sos = if ($dos -gt 0) { "+" } else { "" }
    Write-Host ("{0,-40} {1,9} {2,9} {3,9} {4,9} {5,9}" -f $r.name, ($s50+$dm50), ($s95+$dm95), ($sms+$dms), ($sml+$dml), ($sos+$dos))
}

Write-Host ""
Write-Host "=============================================================================================="
Write-Host "TEST vs VALIDATION CONSISTENCY CHECK"
Write-Host "=============================================================================================="
Write-Host ""
Write-Host ("{0,-45} {1,10} {2,10} {3,10}" -f "Name", "Test_mAP50", "Val_mAP50", "Delta")
Write-Host ("-" * 80)
foreach ($r in $sorted) {
    $v = $validMap[$r.name]
    if ($v) {
        $t = [math]::Round([double]$r.metrics.mAP50_all, 4)
        $va = [math]::Round([double]$v.metrics.mAP50_all, 4)
        $d = [math]::Round($t - $va, 4)
        $sd = if ($d -gt 0) { "+" } else { "" }
        Write-Host ("{0,-45} {1,10} {2,10} {3,10}" -f $r.name, $t, $va, ($sd+$d))
    }
}
