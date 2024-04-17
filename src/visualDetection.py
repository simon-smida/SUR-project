import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ------------------------------------------------------------------

import wandb
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold

# ------------------------------------------------------------------

config = {
    "learning_rate": 0.0001,
    "epochs": 10,
    "batch_size": 32,
    "dropout_rate": 0,
    "batch_norm": "True",
    "architecture": "CNN",
    "dataset": "data"
}

# Format the run name to include key parameters
run_name = f"{config['architecture']}_{config['dataset']}_epochs{config['epochs']}_bn{config['batch_norm']}_lr{config['learning_rate']}_dropout{config['dropout_rate']}"
wandb.init(project='SUR-visual-detection-CV1', entity='xsmida03', name=run_name, config=config)
# ------------------------------------------------------------------


class CNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=wandb.config.dropout_rate):
        super(CNN, self).__init__()
        self.features = nn.Sequential(                   # 80x80 input image
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 40x40 feature map
            nn.Dropout(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 20x20 feature map
            nn.Dropout(dropout_rate)
        )
        
        # Calculate the size of the feature vector
        # Hand-calculated based on the input size and the operations above
        # 64 filters, 20x20 feature map after 2 max pools
        # Needs to be updated if the architecture changes!
        feature_size = 64 * 20 * 20 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, fold_idx=None):
    
    num_epochs = wandb.config.epochs
    best_loss = float('inf')
    no_improve_epoch = 0
    patience = 5
    
    for epoch in tqdm(range(num_epochs), desc='Training', leave=True):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.squeeze().round() == labels).sum().item()
            total_samples += labels.size(0)
            
        # Calculate the average loss and accuracy for the epoch
        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples * 100
        
        # Evaluate the model on the validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, fold_idx)
        
        # Log the metrics 
        log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold_idx)
        tqdm.write(f'[Epoch {epoch+1}/{num_epochs}]\n- Train Loss: {train_loss:.4f}, Train Acc.: {train_accuracy:.2f}%\n- Valid Loss: {val_loss:.4f}, Valid Acc.: {val_accuracy:.2f}%')

        # Early stopping 
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epoch = 0
            # TODO: Save the model
            # torch.save(model.state_dict(), f'best_model_fold_{fold_idx}.pth')
            # wandb.save(f'best_model_fold_{fold_idx}.pth')
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print("Stopping early due to no improvement")
                break

def cross_validation(dataset, num_splits=5):
    
    kfold = KFold(n_splits=num_splits, shuffle=True)
        
    print(f'Performing {num_splits}-fold cross-validation...')
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print('--------------------------------')
        print(f'Fold {fold}/{num_splits}')
        # Prepare the data loaders
        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False)  
        # Initialize the model
        model = CNN(num_classes=1, dropout_rate=wandb.config.dropout_rate)
        optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
        criterion = nn.BCELoss()
        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, fold)
        print('--------------------------------')


def evaluate_model(model, data_loader, criterion, fold_idx=None):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating', leave=True):
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
            total_correct += (outputs.squeeze().round() == labels).sum().item()
            total_samples += labels.size(0)

    # Log validation metrics
    val_loss = total_loss / len(data_loader)
    val_accuracy = total_correct / total_samples * 100
    
    log_validation_accuracy(val_accuracy, fold_idx)   
    return val_loss, val_accuracy

def log_metrics(train_loss, train_accuracy, val_loss, val_accuracy, fold_idx=None):
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

    # Log the metrics to wandb
    wandb.log(metrics)
    
def log_validation_accuracy(val_accuracy, fold_idx=None):
    # Create the key dynamically based on whether fold_idx is provided
    key = "Validation Accuracy" if fold_idx is None else f"Fold {fold_idx+1} Validation Accuracy"
    wandb.log({key: val_accuracy})

def plot_loss_over_epochs(losses, fold=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(f'Loss Over Epochs - Fold {fold}' if fold else 'Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
def save_model(model):
    # Construct unique model name
    config = wandb.config
    name = run_name + '.pth'
    path = os.path.join(os.getcwd(), './trainedModels', name)
    torch.save(model.state_dict(), path)
    wandb.save(path)
    print('Model successfully saved!')

def load_model(model_path):
    model = CNN(num_classes=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_prob = output.item()
    return predicted_prob

def make_decision(probability, threshold=0.5):
    return 'Target' if probability >= threshold else 'Non-target'

def calculate_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    return mean, std


if __name__ == "__main__":
    # Set the base directory and data transforms
    base_dir = os.getcwd() + '/' + wandb.config.dataset
    
    # Pre-calculated mean and standard deviation
    calc_mean = [0.4809, 0.3754, 0.3821]
    calc_std  = [0.2464, 0.2363, 0.2320]

    # Define the data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=calc_mean, std=calc_std)
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=data_transforms)
    dev_dataset = datasets.ImageFolder(os.path.join(base_dir, 'dev'), transform=data_transforms)

    # Concatenate the train and dev datasets = all the data for final training
    dataset = train_dataset #+ dev_dataset
    
    # Initialize the model, loss function, and optimizer
    model = CNN(num_classes=1, dropout_rate=wandb.config.dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Optional: Perform cross-validation
    #cross_validation(dataset, num_splits=5)

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    train_model(model, train_loader, test_loader, criterion, optimizer)
           
    # Save the model
    save_model(model)
    wandb.finish()