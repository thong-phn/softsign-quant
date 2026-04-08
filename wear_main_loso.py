"""
Leave-One-Subject-Out (LOSO) Cross-Validation Script
Iterates through all 21 training subjects.
For each fold:
- Uses 18 subjects for Training [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
- Uses 1 subject for Validation
- Evaluates on the fixed 6 Default Test Subjects [18, 19, 20, 21, 22, 23]
"""
from pathlib import Path
import wandb
import random
import numpy as np
import torch

from lib.wear_train import train_loso
from lib.model import SeparableConvCNN

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse

def main():
    parser = argparse.ArgumentParser(description="LOSO Training script")
    parser.add_argument("--quantization", type=str, choices=['no', 'softsign', 'gamma', 'linear'], default='softsign', help="Quantization layer to use")
    parser.add_argument("--per-channel-quant", action="store_true", help="Use per-channel quantization")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases tracking.")
    args = parser.parse_args()
    quantization = args.quantization
    per_channel_quant = args.per_channel_quant

    set_seed(42)

    project_root = Path(__file__).resolve().parent
    root_path = project_root / "wear"

    # Subjects 0-17 for CV, 18-23 for fixed test
    all_train_subjects = list(range(18))

    # Load fixed Test subjects 18-23
    test_subjects = list(range(18, 24))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Quantization Layer: {quantization}")
    print(f"Using Per-Channel Quantization: {per_channel_quant}")
    print(f"Total Training Subjects ({len(all_train_subjects)}): {all_train_subjects}")
    print(f"Fixed Test Subjects ({len(test_subjects)}): {test_subjects}")
    
    # Store metrics across folds
    fold_results = []

    # Run LOSO (Leave One Subject Out)
    # for val_subject in [0]:
    for val_subject in all_train_subjects:
        val_subjects = [val_subject]
        train_subjects = [s for s in all_train_subjects if s != val_subject]
        
        print(f"\n{'='*50}")
        print(f"Starting Fold | Val Subject: {val_subject} | Train Subjects: {len(train_subjects)}")
        print(f"{'='*50}")

        # Tracking init
        wandb_run = None
        if not args.no_wandb:
            wandb_run = wandb.init(
                project="softsign-quant",
                name=f"wear-val-{val_subject}-quant:{quantization}-per-channel:{per_channel_quant}-{args.run_name}",
                reinit=True, # Allow multiple runs in one script
                config={
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
                },
            )

            # Log code version
            wandb_run.log_code(
                root=str(project_root),
                include_fn=lambda p: p.endswith((".py", ".yaml", ".yml", ".md"))
            )
        
        # Save model dynamically based on fold
        prefix_parts = ["wear_best_model_loso"]
        prefix_parts.append(quantization)
        if per_channel_quant:
            prefix_parts.append("per_channel")
        prefix = "_".join(prefix_parts)
        model_save_path = project_root / "models" / f"{prefix}_val_{val_subject}.pth"
        
        # Define num_channels (assuming 3 for WEAR acc data)
        num_channels = 3

        # Run training loop
        metrics = train_loso(
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

    log_name = "wear_loso_results"
    # if quantization != 'softsign':
    log_name += f"_{quantization}"
    if per_channel_quant:
        log_name += "_per_channel"
    log_name += f"_{args.run_name}.txt"
    
    log_path = log_dir / log_name
    with open(log_path, "w") as f:
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
                gamma_key = "gamma_gamma"
                mu_key = "gamma_mu"
                if gamma_key in metrics and mu_key in metrics:
                    quant_suffix = f" | best gamma = {metrics[gamma_key]} | best mu = {metrics[mu_key]}"

            f.write(
                f"Fold Val {res['val_subject']}: Best Epoch = {best_epoch}, "
                f"Test Acc = {t_acc:.2f}%, Test F1 = {t_f1:.2f}%{quant_suffix}\n"
            )

if __name__ == "__main__":
    import sys
    if "--no-wandb" not in sys.argv:
        wandb.login()
    main()
