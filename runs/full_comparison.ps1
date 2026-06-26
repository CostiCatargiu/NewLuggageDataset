$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__test_full_dataset.json" -Raw | ConvertFrom-Json
$validJson = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\runs_noaug_weapon_full_review\runs_noaug_weapon_full_review__valid_full_dataset.json" -Raw | ConvertFrom-Json

$validMap = @{}
foreach ($r in $validJson.results) { $validMap[$r.name] = $r }

$sorted = @($json.results) | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host "================================================================================================================================"
Write-Host "  FULL DATASET — ALL MODELS COMPARISON"
Write-Host "================================================================================================================================"
Write-Host ""

# Main metrics table
$fmt = "{0,-3} {1,-30} {2,7} {3,7} {4,7} {5,7} {6,7} {7,7} {8,7} {9,7} {10,7}"
Write-Host ($fmt -f "#","Name","T_mAP50","T_m5095","T_Prec","T_Rec","T_mS","T_mM","T_mL","V_mAP50","Mean50")
Write-Host ("-" * 130)

$rank = 1
foreach ($r in $sorted) {
    $m = $r.metrics
    $vm = $validMap[$r.name].metrics
    $mean = [math]::Round(([double]$m.mAP50_all + [double]$vm.mAP50_all) / 2, 4)
    Write-Host ($fmt -f $rank, $r.name, [math]::Round([double]$m.mAP50_all,4), [math]::Round([double]$m.mAP50_95_all,4), [math]::Round([double]$m.precision,4), [math]::Round([double]$m.recall,4), [math]::Round([double]$m.mAP50_small,4), [math]::Round([double]$m.mAP50_medium,4), [math]::Round([double]$m.mAP50_large,4), [math]::Round([double]$vm.mAP50_all,4), $mean)
    $rank++
}

# Per-class AP50 overall
Write-Host ""
Write-Host "================================================================================================================================"
Write-Host "  PER-CLASS AP50 (overall)"
Write-Host "================================================================================================================================"
Write-Host ""
$fmt2 = "{0,-30} {1,8} {2,8} {3,8} {4,8} {5,10}"
Write-Host ($fmt2 -f "Name","knife","long_gun","other","pistol","TAL")
Write-Host ("-" * 80)
foreach ($r in $sorted) {
    $tal = "tk=" + $r.config.tal_topk + " b=" + $r.config.tal_beta
    Write-Host ($fmt2 -f $r.name, [math]::Round([double]$r.per_class.knife.AP50_all,4), [math]::Round([double]$r.per_class.long_gun.AP50_all,4), [math]::Round([double]$r.per_class.other.AP50_all,4), [math]::Round([double]$r.per_class.pistol.AP50_all,4), $tal)
}

# Per-class AP50 small
Write-Host ""
Write-Host "================================================================================================================================"
Write-Host "  PER-CLASS AP50 SMALL"
Write-Host "================================================================================================================================"
Write-Host ""
Write-Host ($fmt2 -f "Name","knife_S","gun_S","other_S","pistol_S","TAL")
Write-Host ("-" * 80)
foreach ($r in $sorted) {
    $tal = "tk=" + $r.config.tal_topk + " b=" + $r.config.tal_beta
    Write-Host ($fmt2 -f $r.name, [math]::Round([double]$r.per_class.knife.AP50_small,4), [math]::Round([double]$r.per_class.long_gun.AP50_small,4), [math]::Round([double]$r.per_class.other.AP50_small,4), [math]::Round([double]$r.per_class.pistol.AP50_small,4), $tal)
}

# Per-class AP50 medium
Write-Host ""
Write-Host "================================================================================================================================"
Write-Host "  PER-CLASS AP50 MEDIUM"
Write-Host "================================================================================================================================"
Write-Host ""
Write-Host ($fmt2 -f "Name","knife_M","gun_M","other_M","pistol_M","TAL")
Write-Host ("-" * 80)
foreach ($r in $sorted) {
    $tal = "tk=" + $r.config.tal_topk + " b=" + $r.config.tal_beta
    Write-Host ($fmt2 -f $r.name, [math]::Round([double]$r.per_class.knife.AP50_medium,4), [math]::Round([double]$r.per_class.long_gun.AP50_medium,4), [math]::Round([double]$r.per_class.other.AP50_medium,4), [math]::Round([double]$r.per_class.pistol.AP50_medium,4), $tal)
}

# Per-class AP50 large
Write-Host ""
Write-Host "================================================================================================================================"
Write-Host "  PER-CLASS AP50 LARGE"
Write-Host "================================================================================================================================"
Write-Host ""
Write-Host ($fmt2 -f "Name","knife_L","gun_L","other_L","pistol_L","TAL")
Write-Host ("-" * 80)
foreach ($r in $sorted) {
    $tal = "tk=" + $r.config.tal_topk + " b=" + $r.config.tal_beta
    Write-Host ($fmt2 -f $r.name, [math]::Round([double]$r.per_class.knife.AP50_large,4), [math]::Round([double]$r.per_class.long_gun.AP50_large,4), [math]::Round([double]$r.per_class.other.AP50_large,4), [math]::Round([double]$r.per_class.pistol.AP50_large,4), $tal)
}

# mAP50-95 per class
Write-Host ""
Write-Host "================================================================================================================================"
Write-Host "  PER-CLASS mAP50-95 (localization quality)"
Write-Host "================================================================================================================================"
Write-Host ""
Write-Host ($fmt2 -f "Name","kn_5095","gn_5095","ot_5095","pi_5095","TAL")
Write-Host ("-" * 80)
foreach ($r in $sorted) {
    $tal = "tk=" + $r.config.tal_topk + " b=" + $r.config.tal_beta
    Write-Host ($fmt2 -f $r.name, [math]::Round([double]$r.per_class.knife.AP50_95_all,4), [math]::Round([double]$r.per_class.long_gun.AP50_95_all,4), [math]::Round([double]$r.per_class.other.AP50_95_all,4), [math]::Round([double]$r.per_class.pistol.AP50_95_all,4), $tal)
}

# Recall per size
Write-Host ""
Write-Host "================================================================================================================================"
Write-Host "  RECALL (AR50) BY SIZE"
Write-Host "================================================================================================================================"
Write-Host ""
$fmt3 = "{0,-30} {1,8} {2,8} {3,8}"
Write-Host ($fmt3 -f "Name","AR50_S","AR50_M","AR50_L")
Write-Host ("-" * 60)
foreach ($r in $sorted) {
    Write-Host ($fmt3 -f $r.name, [math]::Round([double]$r.metrics.AR50_small,4), [math]::Round([double]$r.metrics.AR50_medium,4), [math]::Round([double]$r.metrics.AR50_large,4))
}
