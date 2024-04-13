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
    
    def train(self, X, t, optimizer, loss_function, num_epochs):
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self(X)
            loss = loss_function(output, t)
            loss.backward()
            optimizer.step()
        return loss.item()



if __name__ == '__main__':
    dataPath = os.getcwd() + "/data"
    dirs = ["non_target_train", "target_train","non_target_dev", "target_dev"] 

    non_target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[0])).values())
    target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[1])).values())
    non_target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[2])).values())
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[3])).values())
    
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
    
    # Initialize the MLP
    mlp = MLP()
    optimizer = optim.Adam(mlp.parameters(), lr=0.1)
    loss_function = nn.BCELoss()
    all_loss_lists = []
    all_accuracies = []
    
    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10)
    for train_index, val_index in kf.split(train_dataset):
        train_data, val_data = train_dataset[train_index], train_dataset[val_index]
        
        train_tensor = train_data
        val_tensor = val_data
        # Separate features and target labels
        X_train, t_train = train_tensor[:, :-1], train_tensor[:, -1].reshape(-1, 1)
        X_val, t_val = val_tensor[:, :-1], val_tensor[:, -1].reshape(-1, 1)

        # Initialize a list to store the losses for each epoch
        loss_list = []

        mlp.train(X_train, t_train, optimizer, loss_function, 10)

        # Evaluate the model on the validation set
        val_output = mlp(X_val)
        val_loss = loss_function(val_output, t_val)
        print(f'Validation Loss: {val_loss.item()}')

        # Store the loss list for this fold
        all_loss_lists.append(loss_list)
        
        # Compute accuracy
        predictions = (val_output > 0.5).float()  # Threshold at 0.5
        accuracy = (predictions == t_val).float().mean().item()
        all_accuracies.append(accuracy)

    # Plot the average loss curve across all folds
    average_loss_list = np.mean(all_loss_lists, axis=0)
    plt.plot(average_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Training Loss Curve across 10 Folds')
    plt.show()

    # Plot accuracies across folds
    plt.plot(all_accuracies, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy across 10 Folds')
    plt.grid(True)
    plt.show()

    