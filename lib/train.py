import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from pathlib import Path

# Dataset 
class MyDataset(Dataset):
    def __init__(self, root_path, split='train', subject_ids = None, use_gyro=False):
        """
        Load UCI-HAR data
        Args:
            root_path: Path to dataset
            split: 'train' or 'test'
            subject_ids: List of subject IDs to filter (optional)
            use_gyro: Whether to include gyro (angular velocity) data (default: False)
        """
        self.root_path = Path(root_path)
        self.split_path = self.root_path / split
        self.inertial_path = self.split_path/"Inertial Signals"
        self.use_gyro = use_gyro

        # Load Y (label) and subjects
        path_to_y_file = self.split_path/f"y_{split}.txt"
        path_to_subject_file = self.split_path/f"subject_{split}.txt"

        all_labels = np.loadtxt(path_to_y_file, dtype=int) - 1 # 0-indexed [0, 1, 2, 3, 4, 5]
        all_subjects = np.loadtxt(path_to_subject_file, dtype=int)
        
        # # Load accelerometer data (body acceleration)
        # signal_files = {       
        #     "X": f"body_acc_x_{split}.txt",
        #     "Y": f"body_acc_y_{split}.txt",
        #     "Z": f"body_acc_z_{split}.txt",
        # }
        # Load accelerometer data (body acceleration)
        signal_files = {       
            "X": f"total_acc_x_{split}.txt",
            "Y": f"total_acc_y_{split}.txt",
            "Z": f"total_acc_z_{split}.txt",
        }
        signals = []
        for axis in ["X", "Y", "Z"]:
            data = np.loadtxt(self.inertial_path/signal_files[axis])
            signals.append(data)

        # Load gyro data if requested
        if use_gyro:
            gyro_files = {
                "X": f"body_gyro_x_{split}.txt",
                "Y": f"body_gyro_y_{split}.txt",
                "Z": f"body_gyro_z_{split}.txt",
            }
            for axis in ["X", "Y", "Z"]:
                data = np.loadtxt(self.inertial_path/gyro_files[axis])
                signals.append(data)

        all_signals = np.stack(signals, axis=1) # Stack to shape (samples, num_channels, 128) where num_channels is 3 or 6

        # Return
        if subject_ids is None:
            self.labels = all_labels
            self.signals = all_signals
            self.subjects = all_subjects
        else:
            mask = np.isin(all_subjects, subject_ids)
            self.labels = all_labels[mask]
            self.signals = all_signals[mask]
            self.subjects = all_subjects[mask]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get time-domain signal (3, 128) for accel only, or (6, 128) with gyro
        signal = self.signals[idx]
        
        # Return time domain signal without FFT
        return torch.FloatTensor(signal), torch.LongTensor([self.labels[idx]])[0]
        
# Training function
def train_loso(root_path, model_class, train_subjects, val_subjects, wandb_run=None, **train_kwargs):
    """
    Args:
        root_path: path to UCI-HAR
        model_class
        train_subjects
        val_subjects
        wandb_run:
        **train_kwargs
    """
    # Hyperparameters 
    epochs = train_kwargs.get('epochs', 30)
    lr = train_kwargs.get('lr', 1e-3)
    batch_size = train_kwargs.get('batch_size', 64)
    device = train_kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    patience = train_kwargs.get('patience', 10)
    min_delta = train_kwargs.get('min_delta', 1e-3)
    model_path = Path(train_kwargs.get('model_path', './models/best_model.pth'))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    use_gyro = train_kwargs.get('use_gyro', True)

    # Create dataset and dataloader
    train_dataset = MyDataset(root_path, split='train', subject_ids=train_subjects, use_gyro=use_gyro)
    val_dataset = MyDataset(root_path, split='train', subject_ids=val_subjects, use_gyro=use_gyro)
    test_dataset = MyDataset(root_path, split='test', subject_ids=None, use_gyro=use_gyro)

    # Generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Determine number of channels based on use_gyro
    num_channels = 6 if use_gyro else 3
    print(f"Using {num_channels} channels ({'accel + gyro' if use_gyro else 'accel only'})")

    # Training loop configuration
    quantization = train_kwargs.get('quantization', 'softsign')
    per_channel_quant = train_kwargs.get('per_channel_quant', False)
    model = model_class(num_channels=num_channels, quantization=quantization, per_channel_quant=per_channel_quant).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_epoch = 0 # 
    epochs_no_improve = 0 # early stopping

    print("-"*50)
    # Training loop
    for epoch in range(epochs):
        # Train one epoch
        train_loss_sum = 0.0 # sum of training loss  
        train_correct = 0 # no. of training samples predicted correctly
        train_total = 0 # no. of training samples used
        
        model.train()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # 1. forward 
            loss = criterion(outputs, labels) # 2. loss
            optimizer.zero_grad() # 3. backward: zero_grad
            loss.backward() # cal gradient
            optimizer.step() # update step

            train_loss_sum += loss.item() * labels.size(0) # loss.item() is the average loss of the batch -> recover loss of the batch
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss_sum/train_total
        train_acc = train_correct/train_total * 100.0

        # Val one epoch
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): # no need to track grad in val
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = 100. * val_correct / max(val_total, 1)
        scheduler.step(val_loss) # Step LR scheduler on val loss
        
        # Save best model based on val_loss
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), model_path)
            best_epoch = epoch+1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f'Epoch [{epoch+1}/{epochs}]: '
              f'Train Loss: {train_loss:.4f}; Train Acc: {train_acc:.2f}; '
              f'Val Loss: {val_loss:.4f}; Val Acc: {val_acc:.2f}')

        if hasattr(model, 'quant') and model.quant is not None and quantization in ('softsign', 'linear'):
            k_param = model.quant.k.detach().cpu()
            mu_param = model.quant.mu.detach().cpu()
            if k_param.numel() > 1:
                k_vals = [float(v) for v in k_param.flatten().tolist()]
                mu_vals = [float(v) for v in mu_param.flatten().tolist()]
                print(f"  [{quantization}] epoch params | k={k_vals} | mu={mu_vals}")
            else:
                print(f"  [{quantization}] epoch params | k={float(k_param.item()):.6f} | mu={float(mu_param.item()):.6f}")
        
        if wandb_run is not None: # tracking
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],  # actual LR
                # "epochs_no_improve": epochs_no_improve, # early stopping
            }

            if hasattr(model, 'quant') and model.quant is not None and quantization in ('softsign', 'linear'):
                k_param = model.quant.k.detach().cpu()
                mu_param = model.quant.mu.detach().cpu()
                if k_param.numel() > 1:
                    k_flat = [float(v) for v in k_param.flatten().tolist()]
                    mu_flat = [float(v) for v in mu_param.flatten().tolist()]
                    epoch_log[f"{quantization}_k_mean"] = float(np.mean(k_flat))
                    epoch_log[f"{quantization}_mu_mean"] = float(np.mean(mu_flat))
                    for idx, (k_val, mu_val) in enumerate(zip(k_flat, mu_flat)):
                        epoch_log[f"{quantization}_k_ch{idx}"] = k_val
                        epoch_log[f"{quantization}_mu_ch{idx}"] = mu_val
                else:
                    epoch_log[f"{quantization}_k"] = float(k_param.item())
                    epoch_log[f"{quantization}_mu"] = float(mu_param.item())

            wandb_run.log(epoch_log)

        if epochs_no_improve >= patience: # early stopping
            print(f"Early Stopping: Epoch [{epoch+1}/{epochs}] (patience={patience}, min_delta={min_delta}).")
            break


    # Test with best model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loss_sum, test_correct, test_total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            bs = labels.size(0)
            test_loss_sum += loss.item() * bs
            _, preds = outputs.max(1)
            test_total += bs
            test_correct += preds.eq(labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss_sum / max(test_total, 1)
    test_acc = 100.0 * test_correct / max(test_total, 1)
    test_f1 = f1_score(all_labels, all_preds, average='macro') * 100.0
    
    print("-"*50)
    print(f"Summary:")
    print(f"Best Val Acc: {best_val_accuracy:.2f}% at Epoch {best_epoch}")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | Test F1 (Macro): {test_f1:.2f}%")

    # Extract quantization parameters from best model
    quant_params = {}
    if hasattr(model, 'quant') and model.quant is not None:
        quant_layer = model.quant
        quantization = train_kwargs.get('quantization', 'softsign')
        
        if quantization == 'softsign' or quantization == 'linear':
            # For SoftsignQuant and LinearQuant: extract k and mu
            k_param = quant_layer.k.detach().cpu()
            mu_param = quant_layer.mu.detach().cpu()
            
            # Handle both per-channel and non-per-channel cases
            if k_param.dim() > 0 and k_param.numel() > 1:
                # Per-channel: flatten and convert to list
                quant_params[f"{quantization}_k"] = k_param.flatten().tolist()
                quant_params[f"{quantization}_mu"] = mu_param.flatten().tolist()
            else:
                # Single value
                quant_params[f"{quantization}_k"] = float(k_param.item())
                quant_params[f"{quantization}_mu"] = float(mu_param.item())

            print(f"Best epoch quant params [{quantization}] | k={quant_params[f'{quantization}_k']} | mu={quant_params[f'{quantization}_mu']}")
                
        elif quantization == 'gamma':
            # For GammaQuant: extract gamma and offset (mu)
            gamma_param = quant_layer.gamma_func.gamma.detach().cpu()
            offset_param = quant_layer.gamma_func.offset.detach().cpu()
            
            # Handle both per-channel and non-per-channel cases
            if gamma_param.dim() > 0 and gamma_param.numel() > 1:
                # Per-channel: flatten and convert to list
                quant_params["gamma_gamma"] = gamma_param.flatten().tolist()
                quant_params["gamma_mu"] = offset_param.flatten().tolist()
            else:
                # Single value
                quant_params["gamma_gamma"] = float(gamma_param.item())
                quant_params["gamma_mu"] = float(offset_param.item())

    if wandb_run is not None: # tracking
        summary_dict = {
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_accuracy,
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1_macro": test_f1,
        }
        # Add quantization parameters to summary
        summary_dict.update(quant_params)
        wandb_run.log(summary_dict)

    return {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "model_path": str(model_path),
        **quant_params,
    }




    
