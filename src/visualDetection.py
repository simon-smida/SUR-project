import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# ------------------------------------------------------------------
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold

from utils import log_metrics, calculate_mean_std
# ------------------------------------------------------------------
# TODO: 1. Cross-validation on augmented_data
# TODO: 2. Addressing overfitting (dropout, batch normalization)
# TODO: 3. batchnorm using config
# TODO: 4. Experiments with different datasets and parameters
# TODO: 5. Experiment with test data - being only target
# TODO: 6. Add more augmented data for target, because it has less data than non-target
# ------------------------------------------------------------------

# Argument Parser Setup
parser = argparse.ArgumentParser(description='Train a model with various options')
parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging if set')
parser.add_argument('--cross_validation', action='store_true', help='Perform cross-validation if set')
#args = parser.parse_args()
args, _ = parser.parse_known_args()

# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_NAME = 'SUR-visual-detection'

NS=5                      # Number of splits (folds) for cross-validation
LR=0.0001                 # Learning rate
EPOCHS=50                 # Number of epochs
BATCH_SIZE=64             # Batch size
CNN_DROPOUT_RATE=0.3      # Dropout rate for CNN
FC_DROPOUT_RATE=0.5       # Dropout rate for FC
DATASET='augmented_data'  # Dataset directory (data or augmented_data or augmented_balanced_data)
LOSS='BCEWithLogitsLoss'  # Loss function

config = {
    "loss"          : LOSS,
    "learning_rate" : LR,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "cnn_dropout"   : CNN_DROPOUT_RATE,
    "fc_dropout"    : FC_DROPOUT_RATE,
    "batch_norm"    : "True",
    "architecture"  : "CNN",
    "dataset"       : DATASET
}

# Format the run name to include key parameters
run_name = (
    f"{config['architecture']}_"
    f"{config['loss']}_"
    f"{config['dataset']}_"
    f"epochs{config['epochs']}_"
    f"bs{config['batch_size']}_"
    f"bn{config['batch_norm']}_"
    f"lr{config['learning_rate']}_"
    f"cnnDrop{config['cnn_dropout']}_"
    f"fcDrop{config['fc_dropout']}"
)

# Only initialize wandb if --use_wandb is specified
if args.use_wandb:
    import wandb
    wandb.init(project=PROJECT_NAME, entity='xsmida03', name=run_name, config=config)
# ------------------------------------------------------------------


class CNN(nn.Module):
    def __init__(self, num_classes=1, cnn_dropout=0.3, fc_dropout=0.5):
        super(CNN, self).__init__()
        self.features = nn.Sequential(                   # 80x80 input image
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3 input channels, 32 output feature maps (because of 32 filters) 
            nn.BatchNorm2d(32),                          # Batch normalization 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 40x40 feature map
            nn.Dropout(cnn_dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 20x20 feature map
            nn.Dropout(cnn_dropout)
        )
        
        # Calculate the size of the feature vector
        # NOTE: Hand-calculated based on the input size and the operations above
        # 64 filters, 20x20 feature map after 2 max pools
        # Needs to be updated if the architecture changes!
        feature_size = 64 * 20 * 20 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),   # 128 hidden units
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes),
        )
        
        self.to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, optimizer, criterion, epochs, val_loader=None, fold_idx=None, is_full_training=False):
    """ Train the model on the training set and evaluate on the validation set if provided. """
    best_loss = float('inf')
    no_improve_epoch = 0
    patience = 8
    
    model = model.to(device)

    for epoch in tqdm(range(epochs), desc='Training', leave=True):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = (outputs.squeeze() > 0).float() 
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples * 100
        
        if not is_full_training:
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, fold_idx)
            log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold_idx=fold_idx)
            print(f'- Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'- Valid Loss: {val_loss:.4f}, Valid Acc: {val_accuracy:.2f}%')
            print('------------------------------------------')
            
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
            tqdm.write(f'[Epoch {epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            # Model saving can be based on training loss or omitted for full training

    # Save the model after full training or the last fold of cross-validation
    save_model(model, fold_idx if not is_full_training else None)
    print("-------------------------------------------------------")    
    
def evaluate_model(model, data_loader, criterion, fold_idx=None):
    """ Evaluate the model on the validation set and return the loss and accuracy. """
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating', leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
            probabilities = torch.sigmoid(outputs).squeeze()  # Convert logits to probabilities
            pred = (probabilities > 0.5).float()  # Binarize probabilities to get predictions
            actual = labels.float()
            total_correct += (pred == actual).sum().item()
            total_samples += labels.size(0)
    val_loss = total_loss / len(data_loader)
    val_accuracy = total_correct / total_samples * 100
    return val_loss, val_accuracy

def cross_validation(dataset, num_splits=NS):
    """ Perform k-fold cross-validation on the dataset. """
    kfold = KFold(n_splits=num_splits, shuffle=True)
        
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{num_splits}')
        # Prepare the data loaders
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler   = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_subsampler,   batch_size=BATCH_SIZE, shuffle=False)  
        # Initialize the model
        model = CNN(num_classes=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.86], device=device))
        # Train the model
        train_model(model, train_loader, optimizer, criterion, EPOCHS, val_loader=val_loader, fold_idx=fold)
        print('--------------------------------')

def predict(model, image_tensor):
    """ Predict the probability from the logits output by the model. """
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities
        predicted_prob = probabilities.item()
    return predicted_prob

def make_decision(probability, threshold=0.5):
    """ Make a binary decision based on the probability and threshold. """
    return '1' if probability >= threshold else '0'

def save_model(model, fold_idx=None):
    """ Save the model with a unique name. """
    name = run_name
    if fold_idx is not None:
        name = f"{name}_fold{fold_idx+1}"
    name += '.pth'
    path = os.path.join(os.getcwd(), 'trainedModels', name)
    torch.save(model.state_dict(), path)
    wandb.save(path)
    print(f'Model saved at: {path}')

def load_model(model_path):
    """ Load the model from the specified path. """
    model = CNN(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Print info about the run - full training or cross-validation
def print_run_info(is_cross_validation, train_dataset, dev_dataset):
    print("-------------------------------------------------------")
    if is_cross_validation:
        print(f"Running {NS}-fold Cross-Validation...")
    else:
        print("Running Full Training...")
    print("-------------------------------------------------------")
    print(f"Run name: {run_name}")
    print(f"Dataset: {DATASET}")
    print(f"Train + Dev set size: {len(train_dataset)} + {len(dev_dataset)} = {len(train_dataset) + len(dev_dataset)} (total images)")
    print("-------------------------------------------------------")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Loss function: {LOSS}")
    print(f"Using device: {device}")
    print("-------------------------------------------------------")


if __name__ == "__main__":
    
    # Set the base directory and data transforms
    base_dir = os.getcwd() + '/' + DATASET
    
    # Calculate mean and std on augmented data + original data (both train + dev)
    calc_mean, calc_std = calculate_mean_std(base_dir)
    #print(f"Augmented data (train+dev) mean: {calc_mean}, std: {calc_std}")
    
    # Updated transformations including normalization
    data_transforms = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=calc_mean, std=calc_std)
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=data_transforms)
    dev_dataset   = datasets.ImageFolder(os.path.join(base_dir, 'dev'),   transform=data_transforms)

    # Initialize the model, loss function, and optimizer
    model = CNN(num_classes=1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    DO_CROSS_VALIDATION = args.cross_validation 
    
    if args.cross_validation:
        print_run_info(is_cross_validation=True, train_dataset=train_dataset, dev_dataset=dev_dataset)
        # Combine datasets for cross-validation
        dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])  
        # Perform cross-validation
        cross_validation(dataset, num_splits=NS)
    else:
        print_run_info(is_cross_validation=False, train_dataset=train_dataset, dev_dataset=dev_dataset)
        # Full training on combined dataset without cross-validation
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])
        full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Define weights and criterion for full training
        pos_weight = torch.tensor([0.86], device=device)         # Readjust weight for positive class if needed
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Train the model on the full dataset
        train_model(model, full_loader, optimizer=optimizer, criterion=criterion, epochs=EPOCHS, is_full_training=True)

    wandb.finish()