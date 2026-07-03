#!/bin/bash
set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

MODES=("baseline" "focal" "rare" "both")
SCRIPT="mycode/exp/101_retrain.py"
CONFIG="yelp"

for mode in "${MODES[@]}"; do
    echo "============================================"
    echo "  Running: loss_mode=$mode"
    echo "============================================"

    LOGFILE="$RESULTS_DIR/${mode}.log"

    python "$SCRIPT" -cn "$CONFIG" "loss_mode=$mode" 2>&1 | tee "$LOGFILE"

    echo ""
    echo "  Done: $mode  ->  $LOGFILE"
    echo ""
done

echo "============================================"
echo "  All runs complete. Logs in $RESULTS_DIR/"
echo "============================================"

# Print summary of test metrics from each run
echo ""
echo "====== SUMMARY ======"
for mode in "${MODES[@]}"; do
    echo "--- $mode ---"
    grep -E "f/(trn|val|tst)_(auc|aps|mf1|rec|pre|gme)" "$RESULTS_DIR/${mode}.log" | tail -18
    echo ""
done
