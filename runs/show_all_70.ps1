$json = Get-Content "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\runs_noaug_weapon_70_review__test_full_dataset.json" -Raw | ConvertFrom-Json

$sorted = @($json.results) | Sort-Object {[double]$_.metrics.mAP50_all} -Descending

Write-Host ("Total runs: " + $sorted.Count)
Write-Host ""

$rank = 1
foreach ($r in $sorted) {
    $m = $r.metrics
    $tal = "tk=" + $r.config.tal_topk + "/a=" + $r.config.tal_alpha + "/b=" + $r.config.tal_beta
    Write-Host ($rank.ToString().PadLeft(2) + ". " + $r.name)
    Write-Host ("    mAP50=" + [math]::Round([double]$m.mAP50_all,4) + "  m5095=" + [math]::Round([double]$m.mAP50_95_all,4) + "  Prec=" + [math]::Round([double]$m.precision,4) + "  Rec=" + [math]::Round([double]$m.recall,4) + "  TAL: " + $tal)
    Write-Host ("    small=" + [math]::Round([double]$m.mAP50_small,4) + "  medium=" + [math]::Round([double]$m.mAP50_medium,4) + "  large=" + [math]::Round([double]$m.mAP50_large,4))
    Write-Host ("    knife=" + [math]::Round([double]$r.per_class.knife.AP50_all,4) + "  gun=" + [math]::Round([double]$r.per_class.long_gun.AP50_all,4) + "  other=" + [math]::Round([double]$r.per_class.other.AP50_all,4) + "  pistol=" + [math]::Round([double]$r.per_class.pistol.AP50_all,4))
    Write-Host ("    knife_S=" + [math]::Round([double]$r.per_class.knife.AP50_small,4) + "  gun_S=" + [math]::Round([double]$r.per_class.long_gun.AP50_small,4) + "  other_S=" + [math]::Round([double]$r.per_class.other.AP50_small,4) + "  pistol_S=" + [math]::Round([double]$r.per_class.pistol.AP50_small,4))
    Write-Host ""
    $rank++
}
