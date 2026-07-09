# Stage 3: after headline seeds 6-9 land (46 headline models total),
# run the architecture-robustness check: hidden=1024, headline conditions x 3 seeds.
$gpu = "C:\Users\henri\AppData\Local\Programs\Python\Python313\python.exe"
$exp = "C:\Users\henri\Documents\CommRL\experiments"

while ($true) {
    $n = 0
    foreach ($c in "baseline", "oracle", "learned", "learned_ear") {
        $n += (Get-ChildItem "$exp\results_scout3\${c}_s*\model.pt" -ErrorAction SilentlyContinue).Count
    }
    if ($n -ge 40) { break }   # 4 conds x 10 seeds
    Start-Sleep -Seconds 300
}
Start-Process -FilePath $gpu -ArgumentList "$exp\launch.py", "--script", "train_scout.py", `
    "--outroot", "$exp\results_scout3_wide", "--conditions", "baseline", "oracle", "learned", "learned_ear", `
    "--seeds", "0", "1", "2", "--cycles", "400", "--lam", "0.3", "--voi", "0.2", `
    "--device", "cuda", "--threads", "2", "--hidden", "1024", "--workers", "4" -WindowStyle Hidden
"stage3 wide-model runs launched" | Out-File "$exp\chain_status_stage3.txt"
