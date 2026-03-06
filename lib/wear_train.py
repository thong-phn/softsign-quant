import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path

from lib.wear_data import WearDataset

# Training function
def train_loso(root_path, model_class, train_subjects, val_subjects, test_subjects, wandb_run=None, **train_kwargs):
    """
    Args:
        root_path: path to WEAR
        model_class
        train_subjects
        val_subjects
        test_subjects
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

    # Create dataset and dataloader
    train_dataset = WearDataset(root_path, subject_ids=train_subjects)
    val_dataset = WearDataset(root_path, subject_ids=val_subjects)
    test_dataset = WearDataset(root_path, subject_ids=test_subjects)

    # Generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # WEAR dataset has 3 channels (left_arm_acc_x, y, z) and 8 classes
    num_channels = 3
    num_classes = 8
    print(f"Using {num_channels} channels (accel only) for {num_classes} classes")

    # Training loop configuration
    use_quant = train_kwargs.get('use_quant', True)
    per_channel_quant = train_kwargs.get('per_channel_quant', False)
    
    # We pass num_classes=num_classes down to the model if it supports it
    model = model_class(num_channels=num_channels, num_classes=num_classes, use_quant=use_quant, per_channel_quant=per_channel_quant).to(device)
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
        
        if wandb_run is not None: # tracking
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "lr": optimizer.param_groups[0]["lr"],  # actual LR
                # "epochs_no_improve": epochs_no_improve, # early stopping
            })

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

    if wandb_run is not None: # tracking
        wandb_run.log({
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_accuracy,
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1_macro": test_f1,
        })

    return {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "model_path": str(model_path),
    }




    
