# File        : visualDetection.py
# Date        : 17.4. 2024
# Description : Main script for the visual detection part of the project
# - Perform cross-validation or full training on the image data using a CNN model

import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import argparse
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from imageModel import CNN
from imageConfig import NS, LR, EPOCHS, BATCH_SIZE, DATASET, LOSS, OVERSAMPLING, CNN_DROPOUT_RATE, FC_DROPOUT_RATE
from imageConfig import run_name, config, device
from imageUtils import log_metrics, calculate_mean_std, print_run_info, eval_model
from imageUtils import save_model, cross_validation, train_model, get_class_weights

PROJECT_NAME = 'SUR-visual-detection'


def get_args():
    parser = argparse.ArgumentParser(description='Train a CNN model: cross-validation or full training')
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging if set')
    parser.add_argument('--cross_validation', action='store_true', help='Perform cross-validation if set')
    parser.add_argument('--eval', type=str, help="Path to the model that should be evaluated")
    return parser.parse_known_args()


if __name__ == "__main__":
    
    # Argument parsing
    args, _ = get_args()
    
    # Only initialize wandb if --use_wandb is specified
    if args.use_wandb:
        import wandb
        wandb.init(project=PROJECT_NAME, entity='xsmida03', name=run_name, config=config)
    
    # Set the base directory and data transforms
    base_dir = os.getcwd() + '/' + DATASET
    
    # Calculate mean and std on augmented data + original data (both train + dev)
    calc_mean, calc_std = calculate_mean_std(base_dir)
    
    # Updated transformations including normalization
    data_transforms = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=calc_mean, std=calc_std)
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=data_transforms)
    dev_dataset   = datasets.ImageFolder(os.path.join(base_dir, 'dev'),   transform=data_transforms)
    
    combined_dataset = ConcatDataset([train_dataset, dev_dataset])
    
    # Calculate weights for the combined dataset
    weights = get_class_weights(combined_dataset)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # DataLoader for the combined dataset
    # NOTE: dataloader for OVERSAMPLING method
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)
    # NOTE: Uncomment for NO OVERSAMPLING (weighted loss)
    #combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize the model
    model = CNN(num_classes=1, cnn_dropout=CNN_DROPOUT_RATE, fc_dropout=FC_DROPOUT_RATE)
        
    # Perform cross-validation or full training based on the argument
    if args.cross_validation:
        print_run_info(combined_dataset, is_full_training=False)
        cross_validation(combined_dataset, num_splits=NS)
    elif args.eval:
        eval_model(args.eval, data_transforms)
    else: # Full training
        print_run_info(combined_dataset, is_full_training=True)
        train_model(model, combined_loader, is_full_training=True)
    
    if args.use_wandb:
        wandb.finish()