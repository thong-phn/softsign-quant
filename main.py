"""
Tasks:
	Init model and dataset
	Log training to wandb
"""
from pathlib import Path
import wandb
import random
import numpy as np
import torch
import argparse

from lib.train import train_loso
from lib.model import SeparableConvCNN

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def main():
	# Args
	parser = argparse.ArgumentParser(description="LOSO Training script")
	parser.add_argument("--quantization", type=str, choices=['no', 'softsign', 'gamma', 'linear'], default='softsign')
	parser.add_argument("--per-channel-quant", action="store_true")
	# parser.add_argument("--run_name", type=str)
	# parser.add_argument("--no-wandb", action="store_true")
	args = parser.parse_args()

	# Config
	set_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Load txt file
	project_root = Path(__file__).resolve().parent
	root_path = project_root / "uci-har"
	subject_train_path = root_path / "train" / "subject_train.txt"
	all_subjects = sorted(np.unique(np.loadtxt(subject_train_path, dtype=int)).tolist())
	subject_test_path = root_path / "test" / "subject_test.txt"
	all_test_subjects = sorted(np.unique(np.loadtxt(subject_test_path, dtype=int)).tolist())

	# Val fold
	val_subjects = [1]
	# Train
	train_subjects = [subject for subject in all_subjects if subject not in val_subjects]
	# Test
	test_subjects = [subject for subject in all_test_subjects]

	print(f"Using device: {device}")
	print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
	print(f"Val subjects: {val_subjects}")
	print(f"Test subjects ({len(test_subjects)}): {test_subjects}")

	# Tracking init
	wandb_run = wandb.init(
		project="revised-softsign-quant",
		name=f"[DEBUG]-Q:{args.quantization}-S:{val_subjects}",
		config={
			"train_subjects": train_subjects,
			"val_subjects": val_subjects,
			"test_subjects": test_subjects,
			"epochs": 60,
			"lr": 1e-3,
			"batch_size": 64,
			"model": "SeparableConvCNN",
			"quantization": args.quantization,
			"per_channel_quantization": args.per_channel_quant,
			"use_gyro": True,
		},
	)

	# # Log code version
	# wandb_run.log_code(
	# 	root=str(project_root),
	# 	include_fn=lambda p: p.endswith((".py", ".yaml", ".yml", ".md"))
	# )

	# Run training loop
	metrics = train_loso(
		root_path=root_path,
		model_class=SeparableConvCNN,
		train_subjects=train_subjects,
		val_subjects=val_subjects,
		wandb_run=wandb_run,
		use_gyro=True,
		epochs=60,
		lr=1e-3,
		batch_size=64,
		device=device,
		model_path=project_root / "models" / f"testing_subject{val_subjects[0]}_val.pth",
		quantization=args.quantization,
		per_channel_quant = args.per_channel_quant,
	)

	# Training loop output
	print("Final metrics:")
	for key, value in metrics.items():
		print(f"  {key}: {value}")

	# Tracking finish
	if wandb_run is not None:
		wandb_run.finish()


if __name__ == "__main__":
	wandb.login()
	main()