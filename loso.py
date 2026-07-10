"""
Unified Leave-One-Subject-Out (LOSO) Cross-Validation Script
Supports both UCI-HAR and WEAR datasets.
"""
import argparse
from pathlib import Path
import wandb
import random
import numpy as np
import torch

from lib.train import train_loso as uci_train_loso
from lib.wear_train import train_loso as wear_train_loso
from lib.mhealth_train import train_loso as mhealth_train_loso
from lib.recgym_train import train_loso as recgym_train_loso
from lib.realworld_train import train_loso as realworld_train_loso
from lib.model import SeparableConvCNN

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Unified LOSO Training script")
    parser.add_argument("--dataset", type=str, choices=['uci', 'wear', 'mhealth', 'recgym', 'realworld'], required=True)
    parser.add_argument("--quantization", type=str, choices=['no', 'softsign', 'gamma', 'linear'], default='softsign', help="Quantization layer to use")
    parser.add_argument("--per-channel-quant", action="store_true", help="Use per-channel quantization")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--run_name_prefix", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases tracking.")
    parser.add_argument("--fold", type=int, default=None, help="Optional: specific fold to run (e.g. 0). If not provided, runs all folds.")
    
    args = parser.parse_args()
    dataset = args.dataset
    quantization = args.quantization
    per_channel_quant = args.per_channel_quant
    fold_to_run = args.fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(42)

    project_root = Path(__file__).resolve().parent

    if dataset == 'uci':
        root_path = project_root / "datasets" / "uci-har"
        subject_train_path = root_path / "train" / "subject_train.txt"
        all_train_subjects = sorted(np.unique(np.loadtxt(subject_train_path, dtype=int)).tolist())
        subject_test_path = root_path / "test" / "subject_test.txt"
        test_subjects = sorted(np.unique(np.loadtxt(subject_test_path, dtype=int)).tolist())
    elif dataset == 'wear':
        root_path = project_root / "datasets" / "wear"
        all_train_subjects = list(range(18))
        test_subjects = list(range(18, 24))
        num_channels = 3
    elif dataset == 'recgym':
        root_path = project_root / "datasets" / "recgym"
        all_train_subjects = list(range(1, 9))
        test_subjects = [9, 10]
        num_channels = 6
    elif dataset == 'realworld':
        root_path = project_root / "datasets" / "realworld"
        all_train_subjects = list(range(1, 14))
        test_subjects = [14, 15]
        num_channels = 9
    else:  # mhealth
        root_path = project_root / "datasets" / "mheath"  # directory name is mheath
        all_train_subjects = list(range(1, 10))  # subjects 1-9 for train
        test_subjects = [10]      # subject 10 for test
        num_channels = 9

    print(f"Using device: {device}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Quantization Layer: {quantization}")
    print(f"Using Per-Channel Quantization: {per_channel_quant}")
    print(f"Total Training Subjects ({len(all_train_subjects)}): {all_train_subjects}")
    print(f"Fixed Test Subjects ({len(test_subjects)}): {test_subjects}")
    
    # Store metrics across folds
    fold_results = []

    # Determine which folds to run
    folds = [fold_to_run] if fold_to_run is not None else all_train_subjects
    if fold_to_run is not None and fold_to_run not in all_train_subjects:
        print(f"[Warning] --fold {fold_to_run} is not in the list of training subjects. Exit. ")
        return

    # Run LOSO
    for val_subject in folds:
        val_subjects = [val_subject]
        train_subjects = [s for s in all_train_subjects if s != val_subject]
        
        print(f"\n{'='*50}")
        print(f"Fold (Val Subject): {val_subject} | Train Subjects: {len(train_subjects)}")
        print(f"{'='*50}")

        # Tracking init
        wandb_run = None
        if not args.no_wandb:
            wandb_name = f"[{args.run_name_prefix}]{dataset}-S:{val_subject}-Q:{quantization}-PerC:{per_channel_quant}"
            if args.run_name:
                wandb_name += f"{args.run_name}"
            
            wandb_config = {
                "dataset": dataset,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "epochs": 60,
                "lr": 1e-3,
                "batch_size": 64,
                "model": "SeparableConvCNN",
                "quantization": quantization,
                "per_channel_quant": per_channel_quant,
                "fold": val_subject
            }
            if dataset == 'uci':
                wandb_config["use_gyro"] = True
                
            wandb_run = wandb.init(
                project="revised-softsign-quant",
                name=wandb_name,
                reinit=True,
                config=wandb_config,
            )
        
        # Save model dynamically based on fold
        prefix_parts = [f"{dataset}_best_model_loso", quantization]
        if per_channel_quant:
            prefix_parts.append("per_channel")
        prefix = "_".join(prefix_parts)
        model_save_path = project_root / "models" / f"{prefix}_val_{val_subject}.pth"
        
        # Run training loop
        if dataset == 'uci':
            metrics = uci_train_loso(
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
                model_path=model_save_path,
                quantization=quantization,
                per_channel_quant=per_channel_quant,
            )
        elif dataset == 'wear':
            metrics = wear_train_loso(
                root_path=root_path,
                model_class=SeparableConvCNN,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                wandb_run=wandb_run,
                epochs=60,
                lr=1e-3,
                batch_size=64,
                device=device,
                model_path=model_save_path,
                num_channels=num_channels,
                quantization=quantization,
                per_channel_quant=per_channel_quant
            )
        elif dataset == 'recgym':
            metrics = recgym_train_loso(
                root_path=root_path,
                model_class=SeparableConvCNN,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                wandb_run=wandb_run,
                epochs=60,
                lr=1e-3,
                batch_size=64,
                device=device,
                model_path=model_save_path,
                num_channels=num_channels,
                quantization=quantization,
                per_channel_quant=per_channel_quant
            )
        elif dataset == 'realworld':
            metrics = realworld_train_loso(
                root_path=root_path,
                model_class=SeparableConvCNN,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                wandb_run=wandb_run,
                epochs=60,
                lr=1e-3,
                batch_size=64,
                device=device,
                model_path=model_save_path,
                num_channels=num_channels,
                quantization=quantization,
                per_channel_quant=per_channel_quant
            )
        else: # mhealth
            metrics = mhealth_train_loso(
                root_path=root_path,
                model_class=SeparableConvCNN,
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                wandb_run=wandb_run,
                epochs=60,
                lr=1e-3,
                batch_size=64,
                device=device,
                model_path=model_save_path,
                num_channels=num_channels,
                quantization=quantization,
                per_channel_quant=per_channel_quant
            )

        print(f"\nFinal metrics for Fold (Val {val_subject}):")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
            
        fold_results.append({
            'val_subject': val_subject,
            'metrics': metrics
        })

        if wandb_run is not None:
            wandb_run.finish()

    # Aggregate and print overall LOSO performance
    print(f"\n{'='*50}")
    print("LOSO CROSS-VALIDATION COMPLETE")
    print(f"{'='*50}")
    
    acc_test_list = [res['metrics'].get('test_accuracy', 0) for res in fold_results]
    f1_test_list = [res['metrics'].get('test_f1_macro', 0) for res in fold_results]
    
    avg_test_acc, std_test_acc = np.mean(acc_test_list), np.std(acc_test_list)
    avg_test_f1, std_test_f1 = np.mean(f1_test_list), np.std(f1_test_list)
    
    print(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}%")
    print(f"Average Test F1-Macro: {avg_test_f1:.2f}% ± {std_test_f1:.2f}%")
    
    # Save a log file inside log directory
    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_name = f"{dataset}_loso_results_{quantization}"
    if per_channel_quant:
        log_name += "_per_channel"
    if args.run_name:
        log_name += f"_{args.run_name}"
    log_name += ".txt"
    
    log_path = log_dir / log_name
    with open(log_path, "w") as f:
        f.write(f"Dataset: {dataset.upper()}\n")
        f.write(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {std_test_acc:.2f}%\n")
        f.write(f"Average Test F1-Macro: {avg_test_f1:.2f}% ± {std_test_f1:.2f}%\n\n")
        f.write("Detailed Fold Results:\n")
        for res in fold_results:
            t_acc = res['metrics'].get('test_accuracy', 0)
            t_f1 = res['metrics'].get('test_f1_macro', 0)
            metrics = res['metrics']
            best_epoch = metrics.get('best_epoch', 'N/A')
            quant_suffix = ""

            if quantization in ('softsign', 'linear'):
                k_key = f"{quantization}_k"
                mu_key = f"{quantization}_mu"
                if k_key in metrics and mu_key in metrics:
                    quant_suffix = f" | best k = {metrics[k_key]} | best mu = {metrics[mu_key]}"
            elif quantization == 'gamma':
                if "gamma_gamma" in metrics and "gamma_mu" in metrics:
                    quant_suffix = f" | best gamma = {metrics['gamma_gamma']} | best mu = {metrics['gamma_mu']}"

            f.write(
                f"Fold Val {res['val_subject']}: Best Epoch = {best_epoch}, "
                f"Test Acc = {t_acc:.2f}%, Test F1 = {t_f1:.2f}%{quant_suffix}\n"
            )

if __name__ == "__main__":
    import sys
    if "--no-wandb" not in sys.argv:
        wandb.login()
    main()
