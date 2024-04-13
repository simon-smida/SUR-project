import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold


class VGG(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 40x40 feature map
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # 20x20 feature map
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
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, data_loader, criterion, optimizer, num_epochs):
    epoch_losses = []
    for epoch in tqdm(range(num_epochs), desc='Training', leave=True):
        model.train()
        batch_losses = []
        for images, labels in tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            labels = labels.float()  # Ensure labels are float for BCELoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        tqdm.write(f'Epoch {epoch+1}/{num_epochs} - Average Loss: {epoch_loss:.4f}')
    return epoch_losses

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating', leave=True):
            outputs = model(images)
            predicted = outputs.squeeze().round()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def plot_loss_over_epochs(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_over_epochs.png')
    plt.show()
    
def save_model(model, name):
    path = os.path.join(os.getcwd(), './trainedModels', name)
    torch.save(model.state_dict(), path)
    print('Model saved!')
    
    
def main():
    # Set the base directory and data transforms
    base_dir = os.getcwd() + "/data"
    data_transforms = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=data_transforms)
    dev_dataset = datasets.ImageFolder(os.path.join(base_dir, 'dev'), transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = VGG(num_classes=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 20
    num_splits = 5
    kfold = KFold(n_splits=num_splits, shuffle=True)

    best_accuracy = 0.0
    best_model = None
    best_model_path = None

    dataset = train_dataset
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold+1}/{num_splits}')

        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False)

        model = VGG(num_classes=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, num_epochs)
        accuracy = evaluate_model(model, val_loader)
        print(f'Accuracy for fold {fold+1}: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_path = f'vgg_best_model_fold_{fold+1}.pth'

        print('--------------------------------')

    if best_model:
        save_model(best_model, best_model_path)
        print(f"Best model saved at './trainedModels/{best_model_path}' with accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main()