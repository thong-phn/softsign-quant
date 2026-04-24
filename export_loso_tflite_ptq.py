"""
TFLite Post-Training Quantization for WEAR LOSO models.

Pipeline:  PyTorch .pth  ->  ONNX  ->  TF SavedModel (via onnx2tf)  ->  TFLite (PTQ)
Configs:   W8A16_INT_IO  |  W8A8_INT_IO

Supports:
- WEAR LOSO checkpoints under models/wear/<quantization>/<axis>/...
- UCI-HAR LOSO checkpoints under models/...

Usage:
    python wear_quantize_loso_tflite_ptq.py --dataset wear --subjects '0'
    python wear_quantize_loso_tflite_ptq.py --dataset uci-har --subjects '1'
    python wear_quantize_loso_tflite_ptq.py --subjects '0'
    python wear_quantize_loso_tflite_ptq.py --subjects '0,1,2' --quantization softsign
    python wear_quantize_loso_tflite_ptq.py --subjects '0' \
      --model-path models/wear/softsign/shared-axis/wear_best_model_loso_val_0.pth
"""

import os
import argparse
import copy
from contextlib import contextmanager
import re
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterator
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# Keep TF/TFLite on CPU in environments with older or mismatched CUDA drivers.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#  project imports 
from lib.model import SeparableConvCNN
from lib.train import MyDataset
from lib.wear_data import WearDataset

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


def _parse_subject_selection(subjects_arg: str | None) -> list[int] | None:
    if subjects_arg is None:
        return None
    tokens = [t for t in re.split(r"[\s,\.]+", subjects_arg.strip()) if t]
    if not tokens:
        return None
    return sorted({int(t) for t in tokens})


def _load_subject_ids(path: Path) -> list[int]:
    return sorted(np.unique(np.loadtxt(path, dtype=int)).tolist())


def _make_dataset(
    dataset_name: str,
    root_path: Path,
    subject_ids: list[int],
    split: str | None = None,
):
    if dataset_name == "wear":
        return WearDataset(root_path, subject_ids=subject_ids)

    if dataset_name == "uci-har":
        if split is None:
            raise ValueError("split is required for uci-har datasets")
        return MyDataset(root_path, split=split, subject_ids=subject_ids, use_gyro=True)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


#  Model loading 
def _load_pytorch_model(
    model_path: Path,
    device: torch.device,
    quantization: str,
    per_channel_quant: bool,
    expected_num_channels: int,
):
    """Instantiate/load SeparableConvCNN, then remove learned quant layer for export."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Infer input channels from checkpoint tensors first; this is robust across quant layers.
    if "sep_conv1.depthwise.weight" in state_dict:
        num_channels = int(state_dict["sep_conv1.depthwise.weight"].shape[0])
    elif "quant.k" in state_dict and state_dict["quant.k"].dim() == 3:
        num_channels = int(state_dict["quant.k"].shape[1])
    elif "quant.gamma_func.gamma" in state_dict and state_dict["quant.gamma_func.gamma"].dim() == 3:
        num_channels = int(state_dict["quant.gamma_func.gamma"].shape[1])
    else:
        # Final fallback tied to dataset selection.
        num_channels = int(expected_num_channels)
    
    num_classes  = state_dict["fc2.weight"].shape[0]

    model = SeparableConvCNN(
        num_classes=num_classes,
        num_channels=num_channels,
        quantization=quantization,
        per_channel_quant=per_channel_quant,
    )

    model.load_state_dict(state_dict)

    had_quant_params = any(k.startswith("quant.") for k in state_dict.keys())
    external_preprocessor = None
    if quantization != "no" and hasattr(model, "quant") and model.quant is not None:
        # Keep a frozen copy of the learned input transform and run it outside the exported model.
        external_preprocessor = copy.deepcopy(model.quant)
        external_preprocessor.eval()

    if hasattr(model, "quant"):
        model.quant = torch.nn.Identity()
    model.quantization = "no"

    model.eval()
    return model, num_channels, had_quant_params, external_preprocessor


def _resolve_loso_checkpoint(
    dataset_name: str,
    project_root: Path,
    val_subject: int,
    quantization: str,
    per_channel_quant: bool,
    model_path: Path | None,
):
    if model_path is not None:
        return model_path

    if dataset_name == "uci-har":
        prefix_parts = ["best_model_loso"]
        prefix_parts.append(quantization)
        if per_channel_quant:
            prefix_parts.append("per_channel")
        prefix = "_".join(prefix_parts)
        return project_root / "models" / f"{prefix}_val_{val_subject}.pth"
    
    if dataset_name == "wear":
        prefix_parts = ["wear_best_model_loso"]
        prefix_parts.append(quantization)
        if per_channel_quant:
            prefix_parts.append("per_channel")
        prefix = "_".join(prefix_parts)
        return project_root / "models" / f"{prefix}_val_{val_subject}.pth"


def _count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def _format_ops_macs(ops: float | None, macs: float | None) -> tuple[str, str]:
    ops_str = f"{ops / 1e6:.3f} M" if ops is not None else "N/A"
    macs_str = f"{macs / 1e6:.3f} M" if macs is not None else "N/A"
    return ops_str, macs_str


@contextmanager
def _capture_stderr_bytes() -> Iterator[list[bytes]]:
    """Capture low-level stderr bytes emitted by native code during conversion."""
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    pipe_r, pipe_w = os.pipe()
    os.dup2(pipe_w, stderr_fd)

    captured_chunks: list[bytes] = []
    try:
        yield captured_chunks
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(pipe_w)
        os.close(saved_stderr)

        while True:
            chunk = os.read(pipe_r, 4096)
            if not chunk:
                break
            captured_chunks.append(chunk)
        os.close(pipe_r)


#  ONNX → TF SavedModel 
def _export_onnx_and_convert(
    pt_model: torch.nn.Module,
    onnx_path: Path,
    saved_model_dir: Path,
    input_shape: tuple[int, int, int],
) -> None:
    """Export PyTorch → ONNX → TF SavedModel (via onnx2tf CLI)."""
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        pt_model, dummy, str(onnx_path),
        export_params=True, opset_version=18,
        do_constant_folding=True,
        input_names=["input"], output_names=["output"],
    )
    print(f"  ONNX exported → {onnx_path}")

    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(saved_model_dir), "-osd"]
    print(f"  onnx2tf: {' '.join(cmd)}")
    conv_env = os.environ.copy()
    conv_env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=conv_env,
    )
    print(f"  TF SavedModel → {saved_model_dir}")


#  Representative dataset generator 
def _make_representative_gen(
    dataset: WearDataset,
    n_samples: int = 256,
    input_preprocessor: torch.nn.Module | None = None,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """
    Build a stratified calibration generator that yields float32 numpy arrays
    shaped (1, channels, freq_bins) – one sample at a time.
    ``dataset`` is expected to be a WearDataset instance.
    """
    total = len(dataset)
    if total == 0:
        raise RuntimeError("Empty dataset – cannot build representative set")

    # Collect all labels for stratification
    labels = []
    for i in range(total):
        _, y = dataset[i]
        labels.append(int(y.item()) if isinstance(y, torch.Tensor) else int(y))
    labels = np.array(labels)

    n_samples = min(n_samples, total)

    if n_samples == total:
        indices = np.arange(total)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=42)
        try:
            indices, _ = next(sss.split(np.zeros(total), labels))
        except ValueError:
            rng = np.random.default_rng(42)
            indices = rng.choice(total, size=n_samples, replace=False)

    def gen():
        for idx in indices:
            x, _ = dataset[idx]
            if input_preprocessor is not None:
                with torch.no_grad():
                    x = input_preprocessor(x.unsqueeze(0)).squeeze(0)
            # onnx2tf converts to NHWC: (1, freq_bins, channels)
            arr = x.numpy().astype(np.float32)
            arr = np.transpose(arr)  # (C, F) → (F, C)
            yield [arr[np.newaxis, ...]]

    return gen


#  TFLite conversion 
PTQ_CONFIGS = ["W8A16_INT_IO"]


def _parse_macs_from_log(log_text: str) -> tuple[float | None, float | None]:
    """Extract estimated MACs from TFLite converter log output."""
    # Pattern: "Estimated count of arithmetic ops: 0.574 M  ops, equivalently 0.287 M  MACs"
    match = re.search(
        r"Estimated count of arithmetic ops:\s+([\d.]+)\s+([KMG]?)\s*ops.*?equivalently\s+([\d.]+)\s+([KMG]?)\s*MACs",
        log_text,
    )
    if not match:
        return None, None
    multipliers = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9}
    ops  = float(match.group(1)) * multipliers.get(match.group(2), 1)
    macs = float(match.group(3)) * multipliers.get(match.group(4), 1)
    return ops, macs


def _convert_to_tflite(
    saved_model_dir: str,
    ptq_config: str,
    rep_gen: Callable[[], Iterator[list[np.ndarray]]],
) -> tuple[bytes | None, float | None, float | None]:
    """Convert SavedModel to TFLite with given PTQ config.
    Returns (tflite_bytes, ops, macs) or (None, None, None) on failure."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen

    if ptq_config == "W8A16_INT_IO":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16

    with _capture_stderr_bytes() as captured_chunks:
        try:
            tflite_model = converter.convert()
        except Exception as exc:
            print(f"  [ERROR] TFLite conversion ({ptq_config}): {exc}")
            return None, None, None

    log_text = b"".join(captured_chunks).decode("utf-8", errors="replace")
    ops, macs = _parse_macs_from_log(log_text)
    return tflite_model, ops, macs


#  TFLite evaluation 
def _evaluate_tflite(
    tflite_model: bytes,
    dataloader: DataLoader,
    input_preprocessor: torch.nn.Module | None = None,
) -> tuple[float, float]:
    """Run inference on every sample; return (accuracy%, f1_macro)."""

    # Hide only the known tf.lite.Interpreter deprecation warning from TF 2.20 migration notice.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*Warning: tf\.lite\.Interpreter is deprecated and is scheduled for deletion in\s*TF 2\.20\..*",
            category=UserWarning,
            module=r"tensorflow\.lite\.python\.interpreter",
        )
        interp = tf.lite.Interpreter(model_content=tflite_model)

    interp.allocate_tensors()

    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    inp_idx = inp_det["index"]
    out_idx = out_det["index"]

    is_int_in  = inp_det["dtype"] in (np.int8, np.int16)
    is_int_out = out_det["dtype"] in (np.int8, np.int16)

    in_scale, in_zp   = inp_det["quantization"]
    out_scale, out_zp = out_det["quantization"]
    if in_scale == 0.0:
        in_scale = 1.0
    if out_scale == 0.0:
        out_scale = 1.0

    all_preds, all_targets = [], []

    for x_batch, y_batch in dataloader:
        x_np = x_batch.numpy().astype(np.float32)
        y_np = y_batch.numpy()

        for i in range(len(x_np)):
            sample = x_np[i]                           # (C, F)
            if input_preprocessor is not None:
                sample_t = torch.from_numpy(sample).to(torch.float32)
                with torch.no_grad():
                    sample_t = input_preprocessor(sample_t.unsqueeze(0)).squeeze(0)
                sample = sample_t.numpy()
            # onnx2tf converts to NHWC: (F, C)
            sample = np.transpose(sample)              # (C, F) → (F, C)
            sample = sample[np.newaxis, ...]            # (1, F, C)

            if is_int_in:
                sample = np.round(sample / in_scale + in_zp).astype(inp_det["dtype"])

            interp.set_tensor(inp_idx, sample)
            interp.invoke()
            out = interp.get_tensor(out_idx)[0]

            if is_int_out:
                out = (out.astype(np.float32) - out_zp) * out_scale

            all_preds.append(int(np.argmax(out)))
            all_targets.append(int(y_np[i]))

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    acc = float(np.mean(preds == targets)) * 100.0
    f1  = float(f1_score(targets, preds, average="macro"))
    return acc, f1


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TFLite PTQ evaluation for WEAR LOSO models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wear",
        choices=["wear", "uci-har"],
        help="Dataset to evaluate.",
    )
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated validation subjects, e.g. '0' or '0,1,2'. "
                             "If omitted, runs all subjects.")
    parser.add_argument("--quantization", type=str, default="softsign",
                        choices=["no", "softsign", "gamma", "linear"],
                        help="Quantization layer type used when training the source checkpoint.")
    parser.add_argument("--per-channel-quant", action="store_true",
                        help="Use checkpoints from per-channel runs instead of shared-axis runs.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Optional explicit checkpoint path for single-subject conversion.")
    args = parser.parse_args()

    set_seed(42)

    project_root = Path(__file__).resolve().parent

    if args.dataset == "wear":
        root_path = project_root / "wear"
        all_train_subjects = list(range(18))
        test_eval_subjects = list(range(18, 24))
        test_split = None
        dataset_tag = "wear"
        num_channels = 3
    else:
        root_path = project_root / "uci-har"
        all_train_subjects = _load_subject_ids(root_path / "train" / "subject_train.txt")
        test_eval_subjects = _load_subject_ids(root_path / "test" / "subject_test.txt")
        test_split = "test"
        dataset_tag = "uci-har"
        num_channels = 6

    requested = _parse_subject_selection(args.subjects)
    fold_subjects = requested if requested is not None else all_train_subjects

    if args.model_path is not None and len(fold_subjects) != 1:
        raise ValueError("--model-path can only be used with exactly one subject in --subjects.")

    axis_name = "per-channel" if args.per_channel_quant else "shared-axis"
    results_path = project_root / "log" / f"{dataset_tag}_ptq_results_loso_tflite_{args.quantization}_{axis_name}.txt"
    if args.model_path is not None:
        results_path = project_root / "log" / f"{dataset_tag}_ptq_results_loso_tflite_{args.quantization}_{axis_name}_single.txt"

    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        f.write(f"TFLite PTQ Results - LOSO ({args.quantization}, {axis_name})\n{'=' * 50}\n\n")

    device = torch.device("cpu")          # export & conversion run on CPU
    metrics_history = {c: {"acc": [], "f1": []} for c in PTQ_CONFIGS}

    #  fold loop 
    for val_subject in fold_subjects:
        print(f"\n{'=' * 60}")
        print(f"  Fold – validation subject {val_subject}")
        print(f"{'=' * 60}")

        ckpt = _resolve_loso_checkpoint(
            dataset_name=args.dataset,
            project_root=project_root,
            val_subject=val_subject,
            quantization=args.quantization,
            per_channel_quant=args.per_channel_quant,
            model_path=args.model_path,
        )

        if not ckpt.exists():
            print(f"  [skip] checkpoint not found: {ckpt}")
            continue

        pt_model, in_channels, had_quant_params, input_preprocessor = _load_pytorch_model(
            model_path=ckpt,
            device=device,
            quantization=args.quantization,
            per_channel_quant=args.per_channel_quant,
            expected_num_channels=num_channels,
        )

        #  datasets 
        train_subjects = [s for s in all_train_subjects if s != val_subject]
        train_ds = _make_dataset(args.dataset, root_path, train_subjects, split="train")
        test_ds = _make_dataset(args.dataset, root_path, test_eval_subjects, split=test_split)

        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  [skip] Empty train/test dataset for this fold")
            continue

        # Determine input shape from the first sample
        sample_x, _ = train_ds[0]
        freq_bins = sample_x.shape[-1]
        input_shape = (1, in_channels, freq_bins)
        n_params = _count_parameters(pt_model)
        print(f"  Dataset: {args.dataset}")
        print(f"  Input shape for export: {input_shape}")
        print(f"  PyTorch model parameters: {n_params:,}")
        print(
            "  Export quant layer: disabled "
            f"(checkpoint had quant params: {'yes' if had_quant_params else 'no'})"
        )
        if input_preprocessor is not None:
            print("  Input preprocessing: external learned quant transform is enabled")

        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        #  export & convert once per fold 
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            onnx_path      = tmpdir / "model.onnx"
            saved_model_dir = tmpdir / "saved_model"

            _export_onnx_and_convert(pt_model, onnx_path, saved_model_dir, input_shape)

            rep_gen = _make_representative_gen(
                train_ds,
                n_samples=256,
                input_preprocessor=input_preprocessor,
            )

            # Log model info once per fold
            with open(results_path, "a") as f:
                f.write(
                    f"\nFold {val_subject} | Checkpoint: {ckpt} | Params: {n_params:,}"
                    f" | Quant disabled for export: yes"
                    f" | External input preprocess: {'yes' if input_preprocessor is not None else 'no'}\n"
                )

            #  quantize & eval each config 
            for cfg in PTQ_CONFIGS:
                print(f"\n   {cfg} ")
                tflite_model, ops, macs = _convert_to_tflite(str(saved_model_dir), cfg, rep_gen)
                if tflite_model is None:
                    continue

                tflite_size_kb = len(tflite_model) / 1024

                # Save .tflite for inspection in a persistent location
                tflite_dir = project_root / "models" / "tflite"
                tflite_dir.mkdir(parents=True, exist_ok=True)
                tflite_out = tflite_dir / f"{dataset_tag}_val_{val_subject}_{cfg}.tflite"
                tflite_out.write_bytes(tflite_model)

                ops_str, macs_str = _format_ops_macs(ops, macs)
                print(f"  TFLite size: {tflite_size_kb:.1f} KB  |  OPs: {ops_str}  |  MACs: {macs_str}")

                acc, f1 = _evaluate_tflite(
                    tflite_model,
                    test_loader,
                    input_preprocessor=input_preprocessor,
                )
                print(f"  Result: Acc {acc:.2f}%  F1 {f1:.4f}")

                metrics_history[cfg]["acc"].append(acc)
                metrics_history[cfg]["f1"].append(f1)

                with open(results_path, "a") as f:
                    f.write(
                        f"  {cfg} | Acc: {acc:.2f}% | F1: {f1:.4f}"
                        f" | Size: {tflite_size_kb:.1f} KB"
                        f" | OPs: {ops_str} | MACs: {macs_str}\n"
                    )

    #  summary 
    with open(results_path, "a") as f:
        f.write(f"\n{'=' * 50}\nOVERALL AVERAGES\n{'=' * 50}\n")
        for cfg in PTQ_CONFIGS:
            accs = metrics_history[cfg]["acc"]
            f1s  = metrics_history[cfg]["f1"]
            if accs:
                f.write(f"{cfg}  |  Acc: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%  "
                        f"|  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n")

    print(f"\nResults path: {results_path}")


if __name__ == "__main__":
    main()
