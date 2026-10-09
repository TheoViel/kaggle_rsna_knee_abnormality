#!/bin/bash
# Self-training rounds for the ROI U-Net, run from src/:
#
#   bash roi/iterate.sh 1 3        # rounds 1..3, starting from labels round0
#
# Round r: train on labels round{r-1} -> predict every sagittal / coronal series
# -> merge into labels round{r} (roi.labels merge: U-Net boxes confirmed by the
# other view replace or add labels). Round 0's model is trained separately:
#   torchrun --standalone --nproc_per_node=4 main_roi.py --labels round0
set -e
cd "$(dirname "$0")/.."
FIRST=${1:-1}
LAST=${2:-3}
T="../.venv/bin/torchrun --standalone --nproc_per_node=4 main_roi.py"
latest_run() { ls -d ../logs/roi_run/*/*/ | sort -V | tail -1; }

for r in $(seq "$FIRST" "$LAST"); do
    prev=$((r - 1))
    run=$(latest_run)
    echo "== round $r: predicting with $run (trained on round$prev)"
    $T --predict "$run" --pred_out "round${prev}_pred"
    ../.venv/bin/python -m roi.labels merge --prev "round$prev" --pred "round${prev}_pred" --out "round$r"
    echo "== round $r: training on round$r"
    $T --labels "round$r" --epochs "${EPOCHS:-20}"
done
