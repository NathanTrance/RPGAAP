#!/bin/bash
set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

DATASETS=("yelp" "tfinance" "elliptic" "tolokers")
MODES=("baseline" "focal" "rare")
SCRIPT="mycode/exp/101_retrain.py"

echo "============================================"
echo "  RP-GAAP — 4 datasets x 3 modes"
echo "============================================"
echo ""

for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        LOGFILE="$RESULTS_DIR/${ds}_${mode}.log"
        echo "[$ds] mode=$mode  ->  $LOGFILE"
        python "$SCRIPT" -cn "$ds" "loss_mode=$mode" 2>&1 | tee "$LOGFILE"
        echo ""
    done
done

echo "============================================"
echo "  Done. Logs in $RESULTS_DIR/"
echo "============================================"

echo ""
echo "====== SUMMARY ======"
for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    echo ""
    echo "--- $name ---"
    grep "f/tst_" "$f" | head -10
done
