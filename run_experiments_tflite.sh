#!/bin/bash
# -----------------------------------------------------------------------------
# Automation Script for Softsign Quantization Experiments
# -----------------------------------------------------------------------------

set -e # Exit immediately if a command exits with a non-zero status

# Define arrays for parameters
DATASETS=("uci-har")
# DATASETS=("wear")
QUANT_METHODS=("gamma" "softsign" "linear" "no")
PER_CHANNEL_OPTIONS=("" "--per-channel-quant")
PTQ_SCRIPT="export_loso_tflite_ptq.py"

echo "================================================="
echo "Starting Experiment Pipeline"
echo "================================================="

for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" == "uci-har" ]; then
        MAIN_SCRIPT="main_loso.py"
    else
        MAIN_SCRIPT="wear_main_loso.py"
    fi

    echo "-------------------------------------------------"
    echo "Processing Dataset: $DATASET"
    echo "-------------------------------------------------"

    # 1. Run Shared Axis (no per-channel-quant flag)
    echo ">>> Running Shared Axis Experiments <<<"
    for QUANT in "${QUANT_METHODS[@]}"; do
        echo "Starting Configuration: Dataset=$DATASET | Quantization=$QUANT | Axis=Shared"
        
        echo " -> Training & Evaluation (F32) [Script: $MAIN_SCRIPT]"
        python $MAIN_SCRIPT --quantization $QUANT --run_name SS-scaled-v2 --no-wandb
        
        echo " -> Post-Training Quantization (INT8) [Script: $PTQ_SCRIPT]"
        python $PTQ_SCRIPT --dataset $DATASET --quantization $QUANT
        
        echo "Completed Configuration: $QUANT (Shared)"
        echo "-------------------------------------------------"
    done

    # 2. Run Per-Channel Axis (with per-channel-quant flag)
    echo ">>> Running Per-Channel Axis Experiments <<<"
    for QUANT in "${QUANT_METHODS[@]}"; do
        echo "Starting Configuration: Dataset=$DATASET | Quantization=$QUANT | Axis=Per-Channel"
        
        echo " -> Training & Evaluation (F32) [Script: $MAIN_SCRIPT]"
        python $MAIN_SCRIPT --quantization $QUANT --per-channel-quant --run_name SS-scaled-v2 --no-wandb
        
        echo " -> Post-Training Quantization (INT8) [Script: $PTQ_SCRIPT]"
        python $PTQ_SCRIPT --dataset $DATASET --quantization $QUANT --per-channel-quant
        
        echo "Completed Configuration: $QUANT (Per-Channel)"
        echo "-------------------------------------------------"
    done
done

echo "================================================="
echo "DONE"
echo "================================================="
