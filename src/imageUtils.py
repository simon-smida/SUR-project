# File        : imageUtils.py
# Date        : 17.4. 2024
# Description : Utility functions for the visual detection part of the project

import os

from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import wandb

from imageConfig import NS, LR, EPOCHS, BATCH_SIZE, DATASET, LOSS, CNN_DROPOUT_RATE, FC_DROPOUT_RATE
from imageConfig import run_name, device
from imageModel import CNN


def log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold_idx=None):
    """ Metrics logging for training and validation. """
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
    if val_loss is not None and val_accuracy is not None:
        print(f'- Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'- Valid Loss: {val_loss:.4f}, Valid Acc: {val_accuracy:.2f}%')
        print('------------------------------------------')

def plot_loss_over_epochs(losses, fold=None):
    """ Plot the loss over epochs for a given fold. """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(f'Loss Over Epochs - Fold {fold}' if fold else 'Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
def calculate_mean_std(base_dir):
    """ Calculate the mean and standard deviation of images across all channels in the augmented_data (train+dev). """
    #print("-------------------------------------------------------")    
    initial_transforms = transforms.Compose([
        transforms.Resize((80, 80)), 
        transforms.ToTensor()
    ])

    # Combine datasets for mean/std calculation
    # Create tuples of (image, label) for each image in the dataset
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=initial_transforms)
    dev_dataset   = datasets.ImageFolder(os.path.join(base_dir, 'dev'),   transform=initial_transforms)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])
    # Create a DataLoader to iterate over the combined dataset
    data_loader = DataLoader(combined_dataset, batch_size=128, shuffle=False)
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    # Iterate over the dataset to calculate the mean and standard deviation
    for data, _ in data_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3]) # compute channel-wise (dim 1 are channels, 0: batch, 2, 3: height, width)
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    
    # print(f"Augmented data (train+dev)")
    # print(f"-> mean : {mean}")
    # print(f"-> std  : {std}")
    
    return mean, std

def train_model(model, train_loader, val_loader=None, fold_idx=None, is_full_training=False):
    """ Train the model on the training set and evaluate on the validation set if provided. """
    val_loss, val_accuracy = None, None
    best_loss = float('inf')
    no_improve_epoch = 0
    patience = 5
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    #pos_weight = torch.tensor([0.86]).to(device)
    #pos_weight = torch.tensor([6.4], device=device)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCELoss()
    
    for epoch in tqdm(range(EPOCHS), desc='Training', leave=True):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'[Epoch {epoch+1}/{EPOCHS}]', leave=False):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images).squeeze()                              # Calculate neuron activations
            loss = criterion(outputs, labels.float())  # Compare neuron activations to true labels
            # Backward pass
            loss.backward()  # Calculate the gradients of the loss function w.r.t. the model's parameters
            optimizer.step() # Update the model's parameters based on the gradients
            total_loss += loss.item()
            #predictions = (outputs.squeeze() > 0).float() 
            predictions = torch.round(outputs)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples * 100
        
        if not is_full_training:
            # Evaluate the model on the validation set
            val_loss, val_accuracy = evaluate_model(model, val_loader, fold_idx)
            log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold_idx=fold_idx)
            # Early stopping and model saving based on validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
                if no_improve_epoch >= patience:
                    tqdm.write("Stopping early due to no improvement...")
                    break   
        else: # Full training
            log_metrics(train_loss, train_accuracy, None, None, fold_idx=fold_idx)
            tqdm.write(f'[Epoch {epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

    # Save the model after full training or the last fold of cross-validation
    save_model(model, fold_idx if not is_full_training else None)
    print("-------------------------------------------------------")    
    
    return val_loss, val_accuracy
    
def evaluate_model(model, data_loader, fold_idx=None):
    """ Evaluate the model on the validation set and return the loss and accuracy. """
    model.eval()
    # BCEWithLogitsLoss - use without oversampling
    #pos_weight = torch.tensor([0.86]).to(device)
    #pos_weight = torch.tensor([6.4], device=device)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCELoss()
     
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating', leave=True):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            # Calculate accuracy
            #probabilities = torch.sigmoid(outputs).squeeze()  # Convert logits to probabilities
            #pred = (probabilities > 0.5).float()              # Binarize probabilities to get predictions
            #pred = (outputs.squeeze() > 0).float()
            pred = torch.round(outputs)
            actual = labels.float()
            total_correct += (pred == actual).sum().item()
            total_samples += labels.size(0)
            
    # Calculate the average loss and accuracy
    val_loss = total_loss / len(data_loader)
    val_accuracy = total_correct / total_samples * 100
    return val_loss, val_accuracy

def cross_validation(dataset, num_splits=NS):
    """ Perform k-fold cross-validation on the dataset. """
    kfold = KFold(n_splits=num_splits, shuffle=True)
    accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{num_splits}')
        # Prepare the data loaders
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler   = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_subsampler,   batch_size=BATCH_SIZE, shuffle=False)  
        # Initialize the model
        model = CNN(num_classes=1).to(device)
        # Train the model
        val_loss, val_accuracy = train_model(model, train_loader, val_loader=val_loader, fold_idx=fold)
        accuracies.append(val_accuracy)
        print('--------------------------------')
        
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f'Average validation accuracy: {mean_accuracy:.2f}%')
    if 'wandb' in globals() and wandb.run:
        wandb.log({"Average Validation Accuracy": mean_accuracy})
    return mean_accuracy

def predict(model, image_tensor):
    """ Predict the probability from the logits output by the model. """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        #logits = model(image_tensor)
        #probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        probabilities = model(image_tensor)
        predicted_prob = probabilities.item()
    return predicted_prob

def save_model(model, fold_idx=None):
    """ Save the model with a unique name. """
    name = run_name
    if fold_idx is not None:
        name = f"{name}_fold{fold_idx+1}"
    name += '.pth'
    path = os.path.join(os.getcwd(), 'trainedModels', name)
    torch.save(model.state_dict(), path)
    if wandb.run is not None:
        wandb.save(path)
    print(f'Model saved at: {path}')

def load_model(model_path):
    """ Load the model from the specified path. """
    model = CNN(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def print_run_info(dataset, is_full_training=False):
    """ Print the run information at the start of training. """
    print("-------------------------------------------------------")
    if is_full_training:
        print("Running Full Training...")
    else:
        print(f"Running {NS}-fold Cross-Validation...")
    print("-------------------------------------------------------")
    print(f"Dataset              : {DATASET}")
    print(f"Train + Dev set size : {len(dataset)} (total)")
    print("-------------------------------------------------------")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Droput rates         : CNN={CNN_DROPOUT_RATE}, FC={FC_DROPOUT_RATE}")
    print(f"Learning rate        : {LR}")
    print(f"Epochs               : {EPOCHS}")
    print(f"Loss function        : {LOSS}")
    print(f"Using device         : {device}")
    print("-------------------------------------------------------")
    
def get_class_weights(dataset):
    """ Calculate class weights for the dataset based on the distribution of classes. """
    class_counts = {}
    
    # Check if the dataset is a ConcatDataset and handle accordingly
    if isinstance(dataset, ConcatDataset):
        datasets = dataset.datasets  # Access the list of datasets
    else:
        datasets = [dataset]  # Work with a single dataset as a list

    for ds in datasets:
        if hasattr(ds, 'samples'):
            labels = [label for _, label in ds.samples]
            for label in labels:
                class_counts[label] = class_counts.get(label, 0) + 1
        else:
            raise ValueError("Dataset must have a 'samples' attribute or must be a collection of such datasets")

    total_samples = sum(class_counts.values())
    weight_per_class = {cls: total_samples / float(count) for cls, count in class_counts.items()}

    weights = []
    for ds in datasets:
        weights.extend([weight_per_class[label] for _, label in ds.samples])

    return torch.DoubleTensor(weights)