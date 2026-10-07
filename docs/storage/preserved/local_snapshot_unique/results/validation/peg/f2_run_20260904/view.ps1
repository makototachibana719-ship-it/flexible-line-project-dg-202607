param([string]$View)
$Host.UI.RawUI.WindowTitle = "Peg F2 - $View"
if ($View -eq 'Execution') {
    wsl -d Ubuntu-22.04 -u makoto -- /home/makoto/miniconda3/envs/serl_peg_gpu/bin/python -u /home/makoto/flexible_line_project/results/validation/peg/f2_run_20260904/supervise.py
} else {
    wsl -d Ubuntu-22.04 -u makoto -- tail -n 25 -F /home/makoto/flexible_line_project/results/validation/peg/f2_run_20260904/$View.log
}
