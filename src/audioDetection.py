from scipy.io import wavfile
from glob import glob
import numpy as np
from numpy.random import rand
import scipy
from scipy.fftpack import fft
from numpy.linalg import norm
import matplotlib.pyplot as plt
from numpy import pi
from numpy.random import randint
from scipy.special import logsumexp
from numpy import newaxis
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
from audioDetectionGMM import mel_inv, mel, mel_filter_bank, framing, spectrogram, mfcc, wav16khz2mfcc

class MLP(nn.Module):
    def __init__(self, input_dim=13, layer_width=64, nb_layers=3, nonlinearity=torch.nn.Tanh()):
        super().__init__()
        self.layers = []
        assert nb_layers >= 1

        last_dim = input_dim    #13
        for _ in range(nb_layers):
            self.layers.append(torch.nn.Linear(last_dim, layer_width))
            self.layers.append(nonlinearity)
            last_dim = layer_width
        
        self.layers.append(torch.nn.Linear(last_dim, 1))
        self.layers = torch.nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.layers(x)
        x = nn.Sigmoid()(x)
        return x
    
    def train_model(self, X, t, optimizer, loss_function, num_epochs):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self(X)
            loss = loss_function(output, t)
            loss.backward()
            optimizer.step()
        return loss.item()



if __name__ == '__main__':
    dataPath = os.getcwd() + "/data/train"
    dirs = ["non_target_train", "target_train"]

    non_target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[0])).values())
    target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[1])).values())

    dataPath = os.getcwd() + "/data/dev"
    dirs = ["non_target_dev", "target_dev"] 
    non_target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[0])).values())
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[1])).values())
    
    target_train = np.vstack(target_train)  #konkatenace vsech target_train
    target_dev = np.vstack(target_dev)
    target_train = np.concatenate((target_train, target_dev), axis=0)

    non_target_train = np.vstack(non_target_train)
    non_target_dev = np.vstack(non_target_dev)
    non_target_train = np.concatenate((non_target_train, non_target_dev), axis=0)

    dim = target_train.shape[1]
    target_train = torch.tensor(target_train, dtype=torch.float32)
    target_class = torch.ones(target_train.shape[0], 1)
    target_train = torch.cat((target_train, target_class), dim=1)


    non_target_train = torch.tensor(non_target_train, dtype=torch.float32)
    non_target_class = torch.zeros(non_target_train.shape[0], 1)
    non_target_train = torch.cat((non_target_train, non_target_class), dim=1)

    train_dataset = torch.cat((target_train, non_target_train), dim=0)
    
    # Initialize the KFold object
    kfold = KFold(n_splits=10)

    # Define the number of epochs
    num_epochs = 10

    # Initialize lists to store all losses and accuracies
    all_loss_lists = []
    all_accuracies = []

    # Convert train_dataset to a tensor (assuming train_dataset is a torch tensor)
    train_dataset_tensor = torch.Tensor(train_dataset)
    best_accuracy = 0
    # Perform 10-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset_tensor)):
        print(f'Fold {fold+1}/{kfold.get_n_splits()}')

        # Extract training and validation data tensors
        X_train, t_train = train_dataset_tensor[train_idx, :-1], train_dataset_tensor[train_idx, -1].unsqueeze(1)
        X_val, t_val = train_dataset_tensor[val_idx, :-1], train_dataset_tensor[val_idx, -1].unsqueeze(1)

        # Initialize the model, loss function, and optimizer
        model = MLP()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

        # Train the model
        model.train_model(X_train, t_train, optimizer, criterion, num_epochs)

        # Evaluate the model on the validation set
        val_output = model(X_val)
        val_loss = criterion(val_output, t_val)
        print(f'Validation Loss: {val_loss.item()}')

        # Compute accuracy
        predictions = (val_output > 0.5).float()  # Threshold at 0.5
        accuracy = ((predictions == t_val).float().mean().item())

        print(f'Accuracy for fold {fold+1}: {accuracy:.1%}')

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Store the accuracy
        all_accuracies.append(accuracy)

        print('--------------------------------')

    # Calculate and print the mean accuracy
    mean_accuracy = np.mean(all_accuracies)
    print(f'Mean accuracy: {mean_accuracy:.2f}%')

    # Save the best model
    torch.save(best_model, './trainedModels/audioModelNN.pth')