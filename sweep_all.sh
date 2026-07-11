#!/bin/bash
set -e

RESULTS_DIR="results/sweep_all"
mkdir -p "$RESULTS_DIR"

SCRIPT="mycode/exp/101_retrain.py"

declare -A GPU_MAP=( ["yelp"]=4 ["tfinance"]=5 ["elliptic"]=6 ["tolokers"]=7 )

echo "============================================"
echo "  RP-GAAP Benchmark Sweep — 4 datasets x 12 configs"
echo "============================================"
echo ""

PIDS=()

for ds in "${!GPU_MAP[@]}"; do
    GPU=${GPU_MAP[$ds]}
    (
        run() {
            local cfg="$1" name="$2"
            local log="$RESULTS_DIR/${ds}_${name}.log"
            echo "[GPU $GPU] $ds $name  ->  $log"
            CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" -cn "$ds" $cfg 2>&1 | tee "$log"
            echo ""
        }

        run "loss_mode=baseline"                                                "baseline"
        run "loss_mode=rare"                                                    "rare_default"
        run "loss_mode=rare rare_num_bins=7"                                    "rare_bins7"
        run "loss_mode=rare rare_top_k_features=5"                              "rare_topk5"
        run "loss_mode=rare rare_max_weight=5.0"                                "rare_maxw5"
        run "loss_mode=rare rare_fraud_boost=3.0"                               "rare_boost3"
        run "loss_mode=rare rare_num_bins=7 rare_top_k_features=5"              "rare_bins7_topk5"
        run "loss_mode=rare rare_num_bins=7 rare_top_k_features=5 rare_max_weight=5.0 rare_fraud_boost=3.0" "rare_maxed"
        run "loss_mode=focal"                                                    "focal_default"
        run "loss_mode=focal focal_gamma=3.0"                                   "focal_gamma3"
        run "loss_mode=focal focal_alpha=[1.0,2.0]"                             "focal_alpha_1_2"
        run "loss_mode=focal focal_alpha=[2.0,2.0]"                             "focal_alpha_2_2"

        echo "[GPU $GPU] $ds DONE (12 runs)"
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "============================================"
echo "  Done. Summarizing..."
echo "============================================"

for f in "$RESULTS_DIR"/*.log; do
    bash parse_log.sh "$f"
    echo ""
done
