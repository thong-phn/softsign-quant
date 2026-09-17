# Input Quantization for Wearable Human Activity Recognition, from 4-Bit Sensing to the Binary Limit

[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](Softsign_QUANT_v2_Thong.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

This repository contains the code accompanying **“Input Quantization for Wearable Human Activity Recognition, from 4-Bit Sensing to the Binary Limit.”** The study asks how far inertial sensing precision can be reduced *before* data reaches a wearable host processor, and whether a learnable nonlinear mapping can preserve task-relevant motion information at very low bit depths.

We introduce **Softsign-QUANT**, a learnable sensing-stage quantizer that concentrates resolution around low-amplitude signals using hardware-friendly arithmetic. Across five wearable Human Activity Recognition (HAR) datasets, we compare Softsign-QUANT with learnable Linear and Gamma quantization at 4, 2, and 1 bits under shared-axis and per-axis calibration.

> [!IMPORTANT]
> The paper reports an analytical, instruction-level cycle-energy model—not measurements from deployed firmware. Input quantization reduces sensing and data-movement costs; it does not reduce downstream neural-network arithmetic when sub-8-bit inputs are unpacked to MCU-native integer types.

## Key findings

- **4-bit sensing is a practical operating point.** Across all datasets and configurations, the mean macro-F1 drop from the 12-bit baseline is below one percentage point.
- **Shared-axis calibration is usually sufficient.** It performs close to per-axis calibration while requiring only one learned parameter pair per sensor stream.
- **The binary limit is activity-dependent.** Daily-living activities remain comparatively robust, while sport and gym activities lose amplitude and phase information that is important for recognition.
- **Linear is the lowest-cost baseline.** When nonlinear resolution is useful, Softsign provides a substantially cheaper modeled alternative to Gamma: 29 versus 153 estimated in-sensor cycles per sample under the paper's assumptions.
- **Softsign is the most robust overall at 1 bit.** It limits the worst degradations on fine-grained exercise data and has the smallest average shared-axis F1 drop among the evaluated quantizers.

## Method

For a normalized input sample \(x \in [-1, 1]\), Softsign-QUANT learns a curvature parameter \(k\) and offset \(\mu\):

$$
r(x; k, \mu) = \frac{k(x-\mu)}{1 + |k(x-\mu)|}.
$$

The transformed boundary values are affinely mapped to \([-1, 1]\), then rounded to an unsigned \(b\)-bit grid. A straight-through estimator allows \(k\), \(\mu\), and the classifier to be trained jointly. The affine rescaling uses the complete quantization range even when the learned offset is asymmetric.

![Softsign-QUANT and lightweight 1D CNN architecture](img/model_arch.png)

The classifier is a 9,751-parameter 1D CNN built from depthwise-separable convolutions. The quantizer operates on the input stream before the network, modeling task-aware quantization at the sensing stage.

## Main results

The table below summarizes the paper's shared-axis 4-bit results. Values are macro F1 in percent, averaged over the reported subject folds; the parenthesized value is the change from the 12-bit baseline.

| Dataset | 12-bit baseline | Linear | Softsign | Gamma |
|---|---:|---:|---:|---:|
| WEAR | 69.12 | **68.72 (-0.40)** | 68.24 (-0.88) | 67.98 (-1.14) |
| UCI-HAR | 91.03 | 90.50 (-0.53) | 90.05 (-0.98) | **90.51 (-0.52)** |
| MHEALTH | 67.78 | 67.02 (-0.76) | 69.17 (+1.39) | **69.70 (+1.92)** |
| RealWorld | 65.53 | **66.31 (+0.78)** | 65.28 (-0.25) | 65.50 (-0.03) |
| RecGym | 84.05 | 82.44 (-1.61) | **83.10 (-0.95)** | 82.72 (-1.33) |

These results do not imply that one quantizer wins on every dataset. Linear should remain the first deployment baseline; Softsign is useful when validation accuracy justifies non-uniform resolution, especially at aggressive bit depths.

## Supported datasets

The datasets are not redistributed in this repository. Download them from their original sources and arrange them under `datasets/` as expected by the loaders:

```text
datasets/
├── uci-har/                 # Standard UCI HAR train/ and test/ layout
├── wear/                    # sbj_0.csv ... sbj_23.csv
├── mheath/                  # mHealth_subject1.log ... mHealth_subject10.log
├── recgym/                  # RecGym.csv
└── realworld/
    └── proband1/data/       # Per-activity acc/gyr/mag ZIP files
```

`mheath` is intentionally spelled this way because that is the directory name currently used by the training scripts.

| Dataset | Activity domain | Classes | Channels | Window |
|---|---|---:|---:|---:|
| WEAR | Outdoor exercise | 8 | 3 | 100 samples (2.0 s) |
| UCI-HAR | Daily living | 6 | 6 | 128 samples (2.56 s) |
| MHEALTH | Daily living | 12 | 9 | 100 samples (2.0 s) |
| RealWorld | Daily living | 8 | 9 | 100 samples (2.0 s) |
| RecGym | Gym exercise | 12 | 6 | 40 samples (2.0 s) |

All experiments use 50% overlapping windows. Normalization statistics are fitted on the training fold and then applied to validation and test data.

## Installation

Python 3.10+ and a CUDA-capable PyTorch installation are recommended for full LOSO experiments. CPU execution is supported but considerably slower.

```bash
cd softsign-quant

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pandas scikit-learn wandb
```

The supplied `requirements.txt` is a full snapshot of the development environment, including CUDA, TensorFlow/LiteRT, and nightly conversion packages. It is substantially larger and more platform-specific than the training-only environment above; use it when reproducing the complete export toolchain and adjust platform-specific package references if needed.

## Running experiments

### Train one fold

Use `loso.py` as the unified entry point. Dataset choices are `uci`, `wear`, `mhealth`, `recgym`, and `realworld`.

```bash
# Shared-axis Softsign quantization on one UCI-HAR fold
python loso.py \
  --dataset uci \
  --quantization softsign \
  --fold 1 \
  --no-wandb
```

### Run every configured fold

Omit `--fold` to run the complete set of folds defined for the selected dataset:

```bash
python loso.py --dataset wear --quantization softsign --no-wandb
```

### Compare quantizers and calibration modes

```bash
# Unquantized baseline
python loso.py --dataset recgym --quantization no --no-wandb

# Lowest-cost learnable baseline
python loso.py --dataset recgym --quantization linear --no-wandb

# Gamma-QUANT comparison
python loso.py --dataset recgym --quantization gamma --no-wandb

# Per-axis Softsign parameters instead of one shared pair
python loso.py \
  --dataset recgym \
  --quantization softsign \
  --per-channel-quant \
  --no-wandb
```

Weights & Biases logging is enabled unless `--no-wandb` is supplied. Checkpoints are written to `models/`; fold summaries are written to `log/`.

### Select the sensing precision

The current checkout is configured for the paper's **1-bit experiments** in `SeparableConvCNN` (`lib/model.py`). To reproduce the 4- or 2-bit conditions, set the `bit_width` passed to `SoftsignQuant`, `LinearQuant`, or `GammaQuant` in that constructor to `4` or `2` before launching the run. Keep the bit width identical across methods when making comparisons.

## TFLite post-training quantization

The export utility converts trained checkpoints to integer TFLite models and evaluates them using representative calibration data:

```bash
python export_loso_tflite_ptq.py \
  --dataset uci-har \
  --subjects 1 \
  --quantization softsign

python export_loso_tflite_ptq.py \
  --dataset wear \
  --quantization softsign \
  --per-channel-quant
```

Supported export datasets are `uci-har`, `wear`, `mhealth`, `recgym`, and `realworld`. Use `--model-path` to convert an explicit checkpoint. Exported models are placed in `models/tflite/`, and evaluation reports are placed in `log/`.

TFLite post-training quantization is a supplementary model-export path. It is separate from the paper's sensing-stage bit-depth study and analytical energy model.

## Repository layout

```text
.
├── Softsign_QUANT_v2_Thong.pdf  # Manuscript
├── loso.py                      # Unified fold runner for all five datasets
├── loso-v2.py                   # Nested test/validation LOSO runner for three datasets
├── export_loso_tflite_ptq.py    # TFLite conversion and evaluation
├── lib/
│   ├── model.py                 # CNN and Linear/Softsign/Gamma quantizers
│   ├── train.py                 # UCI-HAR pipeline
│   ├── wear_data.py             # WEAR loader
│   ├── mhealth_data.py          # MHEALTH loader
│   ├── realworld_data.py        # RealWorld loader
│   ├── recgym_data.py           # RecGym loader
│   └── *_train.py               # Dataset-specific training loops
├── img/model_arch.png           # Architecture figure
├── models/                      # Generated checkpoints and TFLite models
└── log/                         # Generated experiment summaries
```

The older dataset-specific entry points are retained for compatibility; new runs should generally use `loso.py`.

## Reproducibility notes

- Random seeds are fixed to 42 and deterministic cuDNN behavior is requested.
- Training uses Adam, a learning rate of `1e-3`, batch size 64, cross-entropy loss, and early stopping.
- Results are reported as macro F1 to account for class imbalance.
- Shared-axis is the default; `--per-channel-quant` enables independent parameters for each sensor channel.
- The energy analysis is comparative and parameterized using assumed instruction counts and datasheet-derived energy-per-cycle values. Absolute latency and energy require measurement on the intended sensor, MCU, compiler, and firmware.

## Citation

The manuscript is currently anonymized. Author, venue, DOI, and archival BibTeX metadata will be added here when the publication record is available. Until then, please cite the title and link to the accompanying manuscript in this repository.

## License

No license file is currently included. Unless a license is added, the source remains under the default copyright protections of its authors.
