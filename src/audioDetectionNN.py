from scipy.io import wavfile
from glob import glob
import numpy as np
from numpy.random import rand
import scipy
from scipy.fftpack import fft
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
from numpy import pi
from numpy.random import randint
from scipy.special import logsumexp
from numpy import newaxis
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold


def mel_inv(x):
    return (np.exp(x/1127.)-1.)*700.


def mel(x):
    return 1127.*np.log(1. + x/700.)


def mel_filter_bank(nfft, nbands, fs, fstart=0, fend=None):
    """Returns mel filterbank as an array (nfft/2+1 x nbands)
    nfft   - number of samples for FFT computation
    nbands - number of filter bank bands
    fs     - sampling frequency (Hz)
    fstart - frequency (Hz) where the first filter strats
    fend   - frequency (Hz) where the last  filter ends (default fs/2)
    """
    if not fend:
      fend = 0.5 * fs

    cbin = np.round(mel_inv(np.linspace(mel(fstart), mel(fend), nbands + 2)) / fs * nfft).astype(int)
    mfb = np.zeros((nfft // 2 + 1, nbands))
    for ii in range(nbands):
        mfb[cbin[ii]:  cbin[ii+1]+1, ii] = np.linspace(0., 1., cbin[ii+1] - cbin[ii]   + 1)
        mfb[cbin[ii+1]:cbin[ii+2]+1, ii] = np.linspace(1., 0., cbin[ii+2] - cbin[ii+1] + 1)
    return mfb


def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def spectrogram(x, window, noverlap=None, nfft=None):
    if np.isscalar(window): window = np.hamming(window)
    if noverlap is None:    noverlap = window.size // 2
    if nfft     is None:    nfft     = window.size
    x = framing(x, window.size, window.size-noverlap)
    x = scipy.fftpack.fft(x*window, nfft)
    return x[:,:x.shape[1]//2+1]

def mfcc(s, window, noverlap, nfft, fs, nbanks, nceps):
    #MFCC Mel Frequency Cepstral Coefficients
    #   CPS = MFCC(s, FFTL, Fs, WINDOW, NOVERLAP, NBANKS, NCEPS) returns 
    #   NCEPS-by-M matrix of MFCC coeficients extracted form signal s, where
    #   M is the number of extracted frames, which can be computed as
    #   floor((length(S)-NOVERLAP)/(WINDOW-NOVERLAP)). Remaining parameters
    #   have the following meaning:
    #
    #   NFFT          - number of frequency points used to calculate the discrete
    #                   Fourier transforms
    #   Fs            - sampling frequency [Hz]
    #   WINDOW        - window lentgth for frame (in samples)
    #   NOVERLAP      - overlapping between frames (in samples)
    #   NBANKS        - numer of mel filter bank bands
    #   NCEPS         - number of cepstral coefficients - the output dimensionality
    #
    #   See also SPECTROGRAM

    # Add low level noise (40dB SNR) to avoid log of zeros 
    snrdb = 40
    noise = rand(s.shape[0])
    s = s + noise.dot(norm(s, 2)) / norm(noise, 2) / (10 ** (snrdb / 20))

    mfb = mel_filter_bank(nfft, nbanks, fs, 32)
    dct_mx = scipy.fftpack.idct(np.eye(nceps, nbanks), norm='ortho') # the same DCT as in matlab

    S = spectrogram(s, window, noverlap, nfft)

    #return np.log(mfb.T.dot(np.abs(S.T))).T
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T

def wav16khz2mfcc(dir_name, feature_len=100):
    """
    Loads all *.wav files from directory dir_name (must be 16kHz), converts them into MFCC 
    features (13 coefficients) and stores them into a dictionary. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    features = {}
    for f in glob(dir_name + '/*.wav'):
        print('Processing file: ', f)
        rate, s = wavfile.read(f)
        assert(rate == 16000)
        features[f] = mfcc(s, 400, 240, 512, 16000, 23, 20)
        

    new_features = [] # each row = 2D array of samesized MFCC reshaped into vector
    for f in features:
        for i in range(features[f].shape[0]):
            if (i+1)*feature_len > features[f].shape[0]: 
                new_features.append(features[f][-feature_len:].reshape(1, -1)[0])
                break
            new_features.append(features[f][i*feature_len:(i+1)*feature_len].reshape(1, -1)[0]) # spectogram matrix to vector
    return new_features


class MLP(nn.Module):
    def __init__(self, input_dim, layer_width=64, nb_layers=3):
        super().__init__()
        self.layers = []
        assert nb_layers >= 1

        last_dim = input_dim
        for _ in range(nb_layers):
            self.layers.append(torch.nn.Linear(last_dim, layer_width))
            self.layers.append(torch.nn.Tanh())
            last_dim = layer_width
        
        self.layers.append(torch.nn.Linear(last_dim, 1))
        self.layers = torch.nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.layers(x)
        x = nn.Sigmoid()(x)
        return x
    
    def train_model(self, X, t, optimizer, loss_function, num_epochs):
        # shuffle the data

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self(X)
            loss = loss_function(output, t)
            loss.backward()
            optimizer.step()
        return loss.item()
    
    def eval(self):
        self.eval()
        return self


def process_data(directory, num_mfcc_features, label):
    # Convert wav files to MFCC features
    mfcc_features = wav16khz2mfcc(directory, num_mfcc_features)
    
    # Convert to NumPy array and then to PyTorch tensor
    mfcc_features = torch.tensor(np.array(mfcc_features), dtype=torch.float32)
    
    # Create class labels tensor
    class_labels = torch.tensor(label * np.ones((mfcc_features.shape[0], 1)), dtype=torch.float32)
    
    # Concatenate features and labels
    mfcc_features_with_labels = torch.cat((mfcc_features, class_labels), dim=1)
    
    return mfcc_features_with_labels

def load_data(window_size=200):
    # Define paths and directories
    train_data_path = os.getcwd() + "/data/train"
    train_directories = ["non_target_train", "target_train"]
    train_augmented_data_path = os.getcwd() + "/augmented_data/train"
    
    # Process training data
    target_train = process_data(os.path.join(train_data_path, train_directories[1]), window_size, 1)
    non_target_train = process_data(os.path.join(train_data_path, train_directories[0]), window_size, 0)

    # Process augmented training data
    target_train_a = process_data(os.path.join(train_augmented_data_path, train_directories[1]), window_size, 1)
    non_target_train_a = process_data(os.path.join(train_augmented_data_path, train_directories[0]), window_size, 0)
    
    
    # Concatenate training and development datasets
    train_dataset = torch.cat((target_train, non_target_train, target_train_a, non_target_train_a), dim=0)

    # ------------------------------------- dev data -------------------------------------
    # Define paths and directories
    dev_data_path = os.getcwd() + "/data/dev"
    dev_directories = ["non_target_dev", "target_dev"]
    dev_augmented_data_path = os.getcwd() + "/augmented_data/dev"
    
    # Process development data
    target_dev = process_data(os.path.join(dev_data_path, dev_directories[1]), window_size, 1)
    non_target_dev = process_data(os.path.join(dev_data_path, dev_directories[0]), window_size, 0)
    
    # Process augmented development data
    target_dev_a = process_data(os.path.join(dev_augmented_data_path, dev_directories[1]), window_size, 1)
    non_target_dev_a = process_data(os.path.join(dev_augmented_data_path, dev_directories[0]), window_size, 0)

    
    # Concatenate training and development datasets
    dev_dataset = torch.cat((target_dev, non_target_dev, target_dev_a, non_target_dev_a), dim=0)

    train_dataset = torch.cat((train_dataset, dev_dataset), dim=0)
    
    return train_dataset


def evaluate_model(train_dataset, num_epochs):
    # Initialize the KFold object
    kfold = KFold(n_splits=10, shuffle=True)

    # Initialize lists to store all losses and accuracies
    all_accuracies = []

    # Perform 10-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold+1}/{kfold.get_n_splits()}')

        # Extract training and validation data tensors
        X_train, t_train = train_dataset[train_idx, :-1], train_dataset[train_idx, -1].unsqueeze(1)
        X_val, t_val = train_dataset[val_idx, :-1], train_dataset[val_idx, -1].unsqueeze(1)

        # Initialize the model, loss function, and optimizer
        model = MLP(input_dim=X_train.shape[1])
        loss = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Train the model
        model.train_model(X_train, t_train, optimizer, loss, num_epochs)
        
        # Evaluate the model on the validation set
        val_output = model(X_val)
        val_loss = loss(val_output, t_val)
        print(f'Validation Loss: {val_loss.item()}')
        # Compute accuracy
        predictions = (val_output > 0.5).float()  # Threshold at 0.5
        accuracy = ((predictions == t_val).float().mean().item())

        print(f'Accuracy for fold {fold+1}: {accuracy:.1%}')

        # Store the accuracy
        all_accuracies.append(accuracy)

        print('--------------------------------')

    print(f'Mean accuracy: {np.mean(all_accuracies):.1%}')



if __name__ == '__main__':
    window_size = 1
    num_epochs = 1000
    dataset = load_data(window_size)
    # K-fold cross-validation
    evaluate_model(dataset, num_epochs)
    """
    # REAL TRAINING BEGINS HERE !!!
    # Initialize the model, loss function, and optimizer
    model = MLP(input_dim=dataset.shape[1])
    loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    num_epochs = 100
    model.train_model(dataset[:, :-1], dataset[:, -1].unsqueeze(1), optimizer, loss, num_epochs)
    model._save_to_state_dict('./trainedModels/model.pth')
    """
    









    

        