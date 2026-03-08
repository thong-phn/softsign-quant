#!/bin/bash
# -----------------------------------------------------------------------------
# Automation Script for Softsign Quantization Experiments
# -----------------------------------------------------------------------------

set -e # Exit immediately if a command exits with a non-zero status
source .venv-litert-torch/bin/activate || echo "Warning: Virtual environment not activated. Ensure dependencies are met."

# Define arrays for parameters
DATASETS=("uci-har" "wear")
QUANT_METHODS=("no" "gamma" "softsign" "linear")
PER_CHANNEL_OPTIONS=("" "--per-channel-quant")

echo "================================================="
echo "Starting Experiment Pipeline"
echo "================================================="

for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" == "uci-har" ]; then
        MAIN_SCRIPT="main_loso.py"
        QUANT_SCRIPT="quantize_loso.py"
    else
        MAIN_SCRIPT="wear_main_loso.py"
        QUANT_SCRIPT="wear_quantize_loso.py"
    fi

    echo "-------------------------------------------------"
    echo "Processing Dataset: $DATASET"
    echo "-------------------------------------------------"

    # 1. Run Shared Axis (no per-channel-quant flag)
    echo ">>> Running Shared Axis Experiments <<<"
    for QUANT in "${QUANT_METHODS[@]}"; do
        echo "Starting Configuration: Dataset=$DATASET | Quantization=$QUANT | Axis=Shared"
        
        echo " -> Training & Evaluation (F32) [Script: $MAIN_SCRIPT]"
        python $MAIN_SCRIPT --quantization $QUANT
        
        echo " -> Post-Training Quantization (INT8) [Script: $QUANT_SCRIPT]"
        python $QUANT_SCRIPT --quantization $QUANT
        
        echo "Completed Configuration: $QUANT (Shared)"
        echo "-------------------------------------------------"
    done

    # 2. Run Per-Channel Axis (with per-channel-quant flag)
    echo ">>> Running Per-Channel Axis Experiments <<<"
    for QUANT in "${QUANT_METHODS[@]}"; do
        echo "Starting Configuration: Dataset=$DATASET | Quantization=$QUANT | Axis=Per-Channel"
        
        echo " -> Training & Evaluation (F32) [Script: $MAIN_SCRIPT]"
        python $MAIN_SCRIPT --quantization $QUANT --per-channel-quant
        
        echo " -> Post-Training Quantization (INT8) [Script: $QUANT_SCRIPT]"
        python $QUANT_SCRIPT --quantization $QUANT --per-channel-quant
        
        echo "Completed Configuration: $QUANT (Per-Channel)"
        echo "-------------------------------------------------"
    done
done

echo "================================================="
echo "All experiments completed successfully!"
echo "Check the 'log/' directory for the detailed results."
echo "================================================="
