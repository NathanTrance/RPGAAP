#!/bin/bash
set -e

RESULTS_DIR="results/sweep"
mkdir -p "$RESULTS_DIR"

DATASET="tfinance"
SCRIPT="mycode/exp/101_retrain.py"

echo "============================================"
echo "  Rare-pattern hyperparameter sweep on $DATASET"
echo "============================================"
echo ""

echo "--- Sweeping rare_num_bins ---"
for bins in 3 5 7; do
    LOGFILE="$RESULTS_DIR/rare_bins${bins}.log"
    echo "rare_num_bins=$bins  ->  $LOGFILE"
    python "$SCRIPT" -cn "$DATASET" loss_mode=rare "rare_num_bins=$bins" 2>&1 | tee "$LOGFILE"
done

echo ""
echo "--- Sweeping rare_top_k_features ---"
for topk in 5 10 15; do
    LOGFILE="$RESULTS_DIR/rare_topk${topk}.log"
    echo "rare_top_k_features=$topk  ->  $LOGFILE"
    python "$SCRIPT" -cn "$DATASET" loss_mode=rare "rare_top_k_features=$topk" 2>&1 | tee "$LOGFILE"
done

echo ""
echo "--- Sweeping rare_fraud_boost ---"
for boost in 1.0 1.5 2.0 3.0; do
    LOGFILE="$RESULTS_DIR/rare_boost${boost}.log"
    echo "rare_fraud_boost=$boost  ->  $LOGFILE"
    python "$SCRIPT" -cn "$DATASET" loss_mode=rare "rare_fraud_boost=$boost" 2>&1 | tee "$LOGFILE"
done

echo ""
echo "--- Sweeping rare_max_weight ---"
for mw in 1.5 2.0 3.0 5.0; do
    LOGFILE="$RESULTS_DIR/rare_maxw${mw}.log"
    echo "rare_max_weight=$mw  ->  $LOGFILE"
    python "$SCRIPT" -cn "$DATASET" loss_mode=rare "rare_max_weight=$mw" 2>&1 | tee "$LOGFILE"
done

echo ""
echo "--- Sweeping focal_alpha ---"
for a1 in 1.0 2.0; do
  for a2 in 1.5 2.0 3.0 4.0; do
    LOGFILE="$RESULTS_DIR/focal_alpha_${a1}_${a2}.log"
    echo "focal_alpha=[$a1, $a2]  ->  $LOGFILE"
    python "$SCRIPT" -cn "$DATASET" loss_mode=focal "focal_alpha=[$a1,$a2]" 2>&1 | tee "$LOGFILE"
  done
done

echo ""
echo "============================================"
echo "  Sweep done. Logs in $RESULTS_DIR/"
echo "============================================"

echo ""
echo "====== SWEEP SUMMARY ======"
for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    echo "--- $name ---"
    grep "f/tst_" "$f" | head -5
    echo ""
done
