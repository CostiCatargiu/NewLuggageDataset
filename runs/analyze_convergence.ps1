$csvDir = "C:\DISK\luggagerepo\NewLuggageDataset\runs\run_weapon_70_review\training_csvs"
$runs = @("r36_p5ctx_seed1_702","r38_gather_704","r38_globalctx_703","r38_dysample_705","r38_bifpn_706","r36_p5big_702","r36_r32b_p5ctx_70")

Write-Host "========================================================================"
Write-Host "TRAINING CURVES: LAST 10 EPOCHS (mAP50 convergence check)"
Write-Host "========================================================================"
Write-Host ""

foreach ($run in $runs) {
    $file = Join-Path $csvDir "$($run)__results.csv"
    if (Test-Path $file) {
        $lines = Get-Content $file
        $totalEpochs = $lines.Count - 1
        
        Write-Host "--- $run ($totalEpochs epochs) ---"
        Write-Host "  Epoch  mAP50       mAP50-95"
        
        $startLine = [math]::Max(1, $lines.Count - 10)
        for ($i=$startLine; $i -lt $lines.Count; $i++) {
            $vals = $lines[$i].Split(",")
            $epoch = $vals[0]
            $m50 = $vals[7]
            $m95 = $vals[8]
            Write-Host ("  {0,5}  {1,10}  {2,10}" -f $epoch, $m50, $m95)
        }
        
        # Best epoch
        $bestM50 = 0
        $bestEp = 0
        for ($i=1; $i -lt $lines.Count; $i++) {
            $vals = $lines[$i].Split(",")
            $v = [double]$vals[7]
            if ($v -gt $bestM50) { $bestM50 = $v; $bestEp = $vals[0] }
        }
        Write-Host ("  BEST: epoch {0}, mAP50 = {1}" -f $bestEp, [math]::Round($bestM50, 5))
        Write-Host ""
    } else {
        Write-Host "--- $run --- FILE NOT FOUND"
        Write-Host ""
    }
}
