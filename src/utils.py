# File: utils.py
# Description: Utility functions for the ML models
# Date: 17.4. 2024

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb 

# TODO: BATCH_SIZE = 128
# TODO: validation stuff

def log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold_idx=None):
    # Check if wandb is initialized
    if 'wandb' in globals() and wandb.run:
        # Base metric keys
        metrics = {
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
        }
        # If fold_idx is specified, prepend it to each metric key
        if fold_idx is not None:
            metrics = {f"Fold {fold_idx+1} {key}": value for key, value in metrics.items()}
        wandb.log(metrics)

# def log_validation_accuracy(val_accuracy, fold_idx=None):
#     if 'wandb' in globals() and wandb.run:
#         # Create the key dynamically based on whether fold_idx is provided
#         key = "Validation Accuracy" if fold_idx is None else f"Fold {fold_idx+1} Validation Accuracy"
#         wandb.log({key: val_accuracy})

def plot_loss_over_epochs(losses, fold=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(f'Loss Over Epochs - Fold {fold}' if fold else 'Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
def calculate_mean_std(base_dir):
    """Calculate the mean and standard deviation of images in the augmented_data (train+dev)."""
    # Initial transformations (for calculating mean/std)
    initial_transforms = transforms.Compose([
        transforms.Resize((80, 80)), 
        transforms.ToTensor()
    ])

    # Combine datasets for mean/std calculation
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=initial_transforms)
    dev_dataset   = datasets.ImageFolder(os.path.join(base_dir, 'dev'),   transform=initial_transforms)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])
    data_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in data_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    return mean, std