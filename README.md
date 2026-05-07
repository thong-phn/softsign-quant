# Softsign Quantization for Wearable HAR

This repository contains the code used to train, evaluate, and export learnable input quantization models for Human Activity Recognition (HAR) on wearable sensor data.

It supports two datasets:

- UCI-HAR: tri-axial accelerometer + gyroscope data
- WEAR: tri-axial accelerometer data from wearable exercise recordings

The main goal is to compare baseline training, learnable quantization layers, and post-training TFLite export for edge deployment.

## Highlights

- Leave-One-Subject-Out (LOSO) training and evaluation
- Learnable quantization layers:
	- `no`
	- `softsign`
	- `gamma`
	- `linear`
- Shared-axis or per-channel quantization
- Optional Weights & Biases logging
- TFLite post-training quantization export and evaluation

## Repository structure

- `main_loso.py` — LOSO training for UCI-HAR
- `wear_main_loso.py` — LOSO training for WEAR
- `export_loso_tflite_ptq.py` — export trained models to TFLite and evaluate PTQ
- `run_experiments_tflite.sh` — automation script for running multiple experiments
- `lib/model.py` — model and quantization layers
- `lib/train.py` — UCI-HAR dataset and training loop
- `lib/wear_data.py` — WEAR dataset loader
- `lib/wear_train.py` — WEAR training loop
- `models/` — saved checkpoints and exported artifacts
- `log/` — experiment summaries

## Requirements

Install the Python dependencies listed in `requirements.txt`.

Recommended environment:

- Python 3.10 or newer
- PyTorch
- TensorFlow / TFLite
- scikit-learn
- Weights & Biases, if logging is enabled

## Data layout

The repository expects the datasets to be available in these folders:

- `uci-har/`
- `wear/`

The UCI-HAR folder should contain the standard train/test splits and inertial signal files. The WEAR folder should contain the subject CSV files already used by the training scripts.

## Training

### UCI-HAR

Train with the default Softsign quantizer:

```bash
python main_loso.py
```

Other options:

```bash
# No quantization
python main_loso.py --quantization no

# Gamma quantization
python main_loso.py --quantization gamma

# Linear quantization
python main_loso.py --quantization linear

# Per-channel quantization
python main_loso.py --per-channel-quant

# Disable Weights & Biases logging
python main_loso.py --no-wandb
```

### WEAR

```bash
python wear_main_loso.py --quantization softsign --run_name my-run
```

Example variations:

```bash
python wear_main_loso.py --quantization no
python wear_main_loso.py --quantization gamma
python wear_main_loso.py --quantization linear
python wear_main_loso.py --per-channel-quant
python wear_main_loso.py --no-wandb
```

## TFLite post-training quantization

After training, export and evaluate a checkpoint with:

```bash
python export_loso_tflite_ptq.py --dataset uci-har --quantization softsign
python export_loso_tflite_ptq.py --dataset wear --quantization softsign
```

Useful options:

```bash
# Evaluate a specific fold
python export_loso_tflite_ptq.py --dataset wear --subjects 0 --quantization softsign

# Use per-channel checkpoints
python export_loso_tflite_ptq.py --dataset uci-har --per-channel-quant --quantization gamma

# Provide an explicit checkpoint path
python export_loso_tflite_ptq.py --dataset wear --subjects 0 \
	--model-path models/wear/softsign/shared-axis/wear_best_model_loso_val_0.pth
```

Exported TFLite files are written to `models/tflite/` and evaluation summaries are written to `log/`.

## Automation

To run the full experiment pipeline used in the project:

```bash
bash run_experiments_tflite.sh
```

This script trains multiple quantization variants and then exports them to TFLite.

## Outputs

- Best PyTorch checkpoints: `models/`
- Exported TFLite models: `models/tflite/`
- LOSO summaries and PTQ reports: `log/`

## Notes

- By default, the training scripts use Weights & Biases tracking.
- Use `--no-wandb` if you want a local-only run.
- The code is designed for LOSO evaluation, so each fold holds out one subject for validation while using the dataset-specific fixed test split.
