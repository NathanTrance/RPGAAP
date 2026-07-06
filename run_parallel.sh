#!/bin/bash
set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

MODES=("baseline" "focal" "rare")
SCRIPT="mycode/exp/101_retrain.py"

# dataset -> GPU mapping
declare -A GPU_MAP=( ["yelp"]=4 ["tfinance"]=5 ["elliptic"]=6 ["tolokers"]=7 )

echo "============================================"
echo "  RP-GAAP — 4 GPUs in parallel"
echo "  GPU 4: yelp | 5: tfinance | 6: elliptic | 7: tolokers"
echo "============================================"
echo ""

PIDS=()

for ds in "${!GPU_MAP[@]}"; do
    GPU=${GPU_MAP[$ds]}
    (
        for mode in "${MODES[@]}"; do
            LOGFILE="$RESULTS_DIR/${ds}_${mode}.log"
            echo "[GPU $GPU] $ds mode=$mode  ->  $LOGFILE"
            CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" -cn "$ds" "loss_mode=$mode" \
                2>&1 | tee "$LOGFILE"
            echo ""
        done
        echo "[GPU $GPU] $ds DONE"
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "============================================"
echo "  All GPUs done. Logs in $RESULTS_DIR/"
echo "============================================"

echo ""
echo "====== SUMMARY ======"
for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    echo ""
    echo "--- $name ---"
    grep "f/tst_" "$f" | head -10
done
