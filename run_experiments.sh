#!/bin/bash
# -----------------------------------------------------------------------------
# Automation Script for Softsign Quantization Experiments
# -----------------------------------------------------------------------------

set -e # Exit immediately if a command exits with a non-zero status

# Define arrays for parameters
DATASETS=("recgym" "realworld" "mhealth" "wear" "uci-har")
QUANT_METHODS=("no" "linear" "softsign" "gamma")
PTQ_SCRIPT="export_loso_tflite_ptq.py"

echo "================================================="
echo "Starting Experiment Pipeline"
echo "================================================="

for DATASET in "${DATASETS[@]}"; do
    MAIN_SCRIPT="loso.py"
    
    # Map dataset name for training script
    if [ "$DATASET" == "uci-har" ]; then
        TRAIN_DATASET="uci"
    else
        TRAIN_DATASET="$DATASET"
    fi

    echo "-------------------------------------------------"
    echo "Processing Dataset: $DATASET (Training as $TRAIN_DATASET)"
    echo "-------------------------------------------------"

    # 1. Run Shared Axis (no per-channel-quant flag)
    echo ">>> Running Shared Axis Experiments <<<"
    for QUANT in "${QUANT_METHODS[@]}"; do
        echo "Starting Configuration: Dataset=$DATASET | Quantization=$QUANT | Axis=Shared"
        
        echo " -> Training & Evaluation (F32) [Script: $MAIN_SCRIPT]"
        python $MAIN_SCRIPT --dataset $TRAIN_DATASET --quantization $QUANT --run_name_prefix 'JUL10,SSV3,NORM'
        
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
        python $MAIN_SCRIPT --dataset $TRAIN_DATASET --quantization $QUANT --per-channel-quant --run_name_prefix '[JUL10,SSV3,NORM]'
        
        echo " -> Post-Training Quantization (INT8) [Script: $PTQ_SCRIPT]"
        python $PTQ_SCRIPT --dataset $DATASET --quantization $QUANT --per-channel-quant
        
        echo "Completed Configuration: $QUANT (Per-Channel)"
        echo "-------------------------------------------------"
    done
done

echo "================================================="
echo "DONE"
echo "================================================="
