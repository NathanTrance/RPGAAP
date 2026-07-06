#!/bin/bash
set -e

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

DATASETS_101=("yelp" "tfinance" "elliptic" "tolokers")
DATASETS_102=("tsocial" "dgraphfin")
MODES=("baseline" "focal" "rare")
SCRIPT_101="mycode/exp/101_retrain.py"
SCRIPT_102="mycode/exp/102_retrain_reallinear_att.py"

echo "============================================"
echo "  RP-GAAP Full Sweep — 6 datasets x 3 modes"
echo "============================================"
echo ""

for ds in "${DATASETS_101[@]}"; do
    for mode in "${MODES[@]}"; do
        LOGFILE="$RESULTS_DIR/${ds}_${mode}.log"
        echo "[$ds] mode=$mode  ->  $LOGFILE"
        python "$SCRIPT_101" -cn "$ds" "loss_mode=$mode" 2>&1 | tee "$LOGFILE"
        echo ""
    done
done

for ds in "${DATASETS_102[@]}"; do
    for mode in "${MODES[@]}"; do
        LOGFILE="$RESULTS_DIR/${ds}_${mode}.log"
        echo "[$ds] mode=$mode  ->  $LOGFILE"
        python "$SCRIPT_102" -cn "$ds" "loss_mode=$mode" 2>&1 | tee "$LOGFILE"
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
