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
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T


def wav16khz2mfcc(dir_name):
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
        features[f] = mfcc(s, 400, 240, 512, 16000, 23, 13)
    return features


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



if __name__ == '__main__':
    dataPath = os.getcwd() + "/data"
    dirs = ["non_target_train", "target_train","non_target_dev", "target_dev"] 

    non_target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[0])).values())
    target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[1])).values())
    non_target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[2])).values())
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[3])).values())
    
    target_train = np.vstack(target_train)  #konkatenace vsech target_train
    non_target_train = np.vstack(non_target_train)

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
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    loss_function = nn.BCELoss()
    all_loss_lists = []
    
    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10)
    for train_index, val_index in kf.split(train_dataset):
        train_data, val_data = train_dataset[train_index], train_dataset[val_index]

        # Convert the training and validation data back to tensors
        train_tensor = train_data
        val_tensor = val_data
        # Separate features and target labels
        X_train, t_train = train_tensor[:, :-1], train_tensor[:, -1].reshape(-1, 1)
        X_val, t_val = val_tensor[:, :-1], val_tensor[:, -1].reshape(-1, 1)

        # Initialize a list to store the losses for each epoch
        loss_list = []

        # Train the MLP
        for epoch in tqdm(range(100), desc='Epoch'):
            optimizer.zero_grad()
            output = mlp(X_train)
            loss = loss_function(output, t_train)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # Evaluate the model on the validation set
        val_output = mlp(X_val)
        val_loss = loss_function(val_output, t_val)
        print(f'Validation Loss: {val_loss.item()}')

        # Store the loss list for this fold
        all_loss_lists.append(loss_list)

    # Plot the average loss curve across all folds
    average_loss_list = np.mean(all_loss_lists, axis=0)
    plt.plot(average_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Training Loss Curve across 10 Folds')
    plt.show()
    