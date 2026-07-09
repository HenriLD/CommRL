# Fills GPU capacity as sweep queues drain:
# 1) when the secondary-condition queue finishes -> lambda x VOI grid (4 workers)
# 2) when the main headline queue finishes -> seeds 6-9 for headline conditions (3 workers)
$gpu = "C:\Users\henri\AppData\Local\Programs\Python\Python313\python.exe"
$exp = "C:\Users\henri\Documents\CommRL\experiments"

function CountModels($pattern) {
    (Get-ChildItem $pattern -ErrorAction SilentlyContinue).Count
}

# wait for secondary queue (18 runs across 6 conditions x 3 seeds)
while ($true) {
    $n = 0
    foreach ($c in "simple", "exclusivity", "progress", "filter", "learned_prag", "filter_ear") {
        $n += CountModels "$exp\results_scout3\${c}_s*\model.pt"
    }
    if ($n -ge 18) { break }
    Start-Sleep -Seconds 300
}
foreach ($pt in @(@("0.1", "0.2"), @("0.6", "0.2"), @("1.0", "0.2"), @("0.3", "0.0"))) {
    $out = "$exp\results_scout3_grid\lam$($pt[0])_voi$($pt[1])"
    Start-Process -FilePath $gpu -ArgumentList "$exp\launch.py", "--script", "train_scout.py", `
        "--outroot", $out, "--conditions", "learned_ear", "--seeds", "0", "1", "2", `
        "--cycles", "400", "--lam", $pt[0], "--voi", $pt[1], "--device", "cuda", `
        "--threads", "2", "--workers", "1" -WindowStyle Hidden
}
"grid launched" | Out-File "$exp\chain_status.txt"

# wait for main headline queue (5 conditions x 6 seeds)
while ($true) {
    $n = 0
    foreach ($c in "baseline", "oracle", "learned", "learned_ear", "ear") {
        $n += CountModels "$exp\results_scout3\${c}_s*\model.pt"
    }
    if ($n -ge 30) { break }
    Start-Sleep -Seconds 300
}
Start-Process -FilePath $gpu -ArgumentList "$exp\launch.py", "--script", "train_scout.py", `
    "--outroot", "$exp\results_scout3", "--conditions", "baseline", "oracle", "learned", "learned_ear", `
    "--seeds", "6", "7", "8", "9", "--cycles", "400", "--lam", "0.3", "--voi", "0.2", `
    "--device", "cuda", "--threads", "2", "--workers", "3" -WindowStyle Hidden
"grid + extra seeds launched" | Out-File "$exp\chain_status.txt"
