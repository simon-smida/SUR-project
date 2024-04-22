from scipy.io import wavfile
from glob import glob
import numpy as np
from numpy.random import rand
import scipy
from numpy.linalg import norm
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import sys
#import wandb


config={
    "learning_rate": 0.0001,
    "epochs": 100,
    "batch_size": 64,
    "dropout_rate": 0.2,
    "window_size": 400,
    "architecture": "MLP",
    "dataset": "data"
}

"""
run_name = f"{config['architecture']}_{config['dataset']}_epochs{config['epochs']}_ws{config['window_size']}_lr{config['learning_rate']}_dropout{config['dropout_rate']}_bs{config['batch_size']}"
wandb.init(project='SUR-audio-detection', entity='maxim-pl', name=run_name, config=config)
"""


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

    #return (np.log(mfb.T.dot(np.abs(S.T)))).T
    return dct_mx.dot(np.log(mfb.T.dot(np.abs(S.T)))).T


def wav16khz2mfcc(dir_name, class_label):
    """
    Loads all *.wav files from directory dir_name (must be 16kHz), converts them into MFCC 
    features (13 coefficients) and stores them into a dictionary. Keys are the file names
    and values and 2D numpy arrays of MFCC features.
    """
    window_size = config['window_size']
    features = []
    for f in glob(dir_name + '/*.wav'):
        #print('Processing file: ', f)
        rate, s = wavfile.read(f)
        assert(rate == 16000)
        result = mfcc(s, 400, 240, 512, 16000, 23, 20)
        new_features = makeWindowedData(result, window_size, class_label)
        features.append(new_features)
    return features


def makeWindowedData(features, window_size, class_label):
    num_of_samples = window_size * 20 # 20 MFCC coefficients, 100 window_size = cca 1 second
    features = features.flatten() # 2D array to 1D array
    
    new_features = [] # each row = num_of_samples
    for i in range(len(features)):
        if (i+1)*num_of_samples > len(features): 
            new_features.append(features[-num_of_samples:])
            break
        new_features.append(features[i*num_of_samples:(i+1)*num_of_samples])
    
    
    features = torch.tensor(np.array(new_features), dtype=torch.float32)
    labels = torch.tensor([class_label]*len(features), dtype=torch.long)
    features = torch.cat((features, labels.unsqueeze(1)), 1)
    
    return features


def load_data():
    dirs_target = ['augmented_data/train/target_train', 'augmented_data/dev/target_dev']
    dirs_non_target = ['augmented_data/train/non_target_train', 'augmented_data/dev/non_target_dev']
    dataset = []  # 0 - non_target, 1 - target
    
    for d in dirs_target:
        dataset.extend(wav16khz2mfcc(d, 1)) 

    for d in dirs_non_target:
        dataset.extend(wav16khz2mfcc(d, 0))

    return dataset



class MLP(nn.Module):
    def __init__(self, input_dim, layer_width=32):
        super().__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(input_dim, layer_width))
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(torch.nn.Dropout(p=config['dropout_rate']))
        self.layers.append(torch.nn.Linear(layer_width, layer_width))
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(torch.nn.Dropout(p=config['dropout_rate']))
        self.layers.append(torch.nn.Linear(layer_width, layer_width))
        self.layers.append(torch.nn.Sigmoid())
        self.layers.append(torch.nn.Dropout(p=config['dropout_rate']))
        
        self.layers.append(torch.nn.Linear(layer_width, 1))
        self.layers = torch.nn.Sequential(*self.layers)


    def forward(self, x):
        x = self.layers(x)
        return x

    def train_model(self, X, t, optimizer, loss_function):
        num_epochs = config['epochs']
        batch_size = config['batch_size']
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        best_loss = float('inf')
        patience = 10
            
        loss_array = []
        accuracy_array = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0

            # Shuffle data for each epoch
            indices = torch.randperm(num_samples)
            X_shuffled = X[indices]
            t_shuffled = t[indices]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                t_batch = t_shuffled[start_idx:end_idx]

                optimizer.zero_grad()
                output_batch = self(X_batch)
                loss = loss_function(output_batch, t_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Compute accuracy for this batch
                predicted_labels = (output_batch > 0.5).float()
                batch_accuracy = (predicted_labels == t_batch).float().mean()
                epoch_accuracy += batch_accuracy.item()

            # Average loss and accuracy over all batches in the epoch
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches

            loss_array.append(epoch_loss)
            accuracy_array.append(epoch_accuracy)
            #wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_accuracy})
            print(f'Epoch {epoch}: Loss = {epoch_loss}, Accuracy = {epoch_accuracy}')
            
            # Early stopping 
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 10
                best_model = self.state_dict()
            else:
                patience -= 1
                if patience == 0:
                    print("Stopping early due to no improvement")
                    self.load_state_dict(best_model)
                    break

        return loss_array, accuracy_array


def evaluate_model(train_dataset):

    # Initialize the KFold object
    kfold = KFold(n_splits=10, shuffle=True)

    # Initialize lists to store all losses and accuracies
    all_accuracies = []

    # Perform 10-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold}')

        # Split the dataset into training and validation sets
        train_set = torch.cat([train_dataset[i] for i in train_idx])
        val_set = torch.cat([train_dataset[i] for i in val_idx])

        X_train, t_train = train_set[:, :-1], train_set[:, -1].unsqueeze(1)
        X_val, t_val = val_set[:, :-1], val_set[:, -1].unsqueeze(1)

        # Initialize the model, loss function, and optimizer
        model = MLP(input_dim=X_train.shape[1])
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.86]))
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Train the model
        model.train_model(X_train, t_train, optimizer, loss)
        
        model.eval()
        # Evaluate the model on the validation set
        val_output = model(X_val)
        s_val_output = nn.Sigmoid()(val_output)
        val_loss = loss(s_val_output, t_val)
        # Compute accuracy
        predictions = (s_val_output > 0.5).float()  # Threshold at 0.5
        
        val_accuracy = ((predictions == t_val).float().mean().item())
        #wandb.log({"Validation Loss": val_loss.item(),"Validation Accuracy": val_accuracy})
        print(f'Validation accuracy: {val_accuracy}')
        # Store the accuracy
        all_accuracies.append(val_accuracy)

    #wandb.log({"Mean accuracy": np.mean(all_accuracies)})
    print(f'Mean accuracy: {np.mean(all_accuracies)}')

def train_final(dataset):
    # Use whole dataset for final training
    train_set = torch.cat(dataset)
    X_train, t_train = train_set[:, :-1], train_set[:, -1].unsqueeze(1)

    # Initialize the model, loss function, and optimizer
    model = MLP(input_dim=X_train.shape[1])
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.86]))
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train the model
    model.train_model(X_train, t_train, optimizer, loss)
    # Initialize the model, loss function, and optimizer
    torch.save(model.state_dict(), './trainedModels/audioModelNN.pth')
    #print('Model saved')


def load_final():
    window_size = config['window_size']
    features = {}
    test_dir = 'data/test'
    for f in glob(test_dir + '/*.wav'):
        rate, s = wavfile.read(f)
        assert rate == 16000

        # Determine the number of samples corresponding to 2 seconds
        num_samples_to_trim = rate * 2
        
        # Check if the audio file is longer than 2 seconds
        if len(s) > num_samples_to_trim:
            # Trim the first 2 seconds of audio
            s_trimmed = s[num_samples_to_trim:]

            # Extract features from the trimmed audio
            result = mfcc(s_trimmed, 400, 240, 512, 16000, 23, 20)
            new_features = makeWindowedData(result, window_size, 0)

            features[f] = new_features
        else:
            print(f"Skipping {f} as it is shorter than 2 seconds.")

    return features

def final_test(test_dataset):
    model = MLP(input_dim=test_dataset[list(test_dataset.keys())[0]].shape[1]-1)
    model.load_state_dict(torch.load('./trainedModels/audioModelNN.pth'))
    model.eval()
    for f in test_dataset:
        file_name = f.split('/')[-1]
        X_test = test_dataset[f][:, :-1] # remove labels
        test_output = model(X_test)
        test_output = nn.Sigmoid()(test_output) # convert to probability
        output = (test_output).mean().item() # model output is logits, convert to probability
        print(f'{file_name[:-4]} {output:.2f} {1 if output > 0.5 else 0}')


if __name__ == '__main__':

    if '--train' in sys.argv:
        train = True
    else:
        train = False

    if '--evaluate' in sys.argv:
        evaluate = True
    else:
        evaluate = False

    if train:
        dataset = load_data() # list of files, each element is 2D numpy array of MFCC features connected 
        if evaluate:
            evaluate_model(dataset)
        else:
            train_final(dataset)
    else:
        test_dataset = load_final()
        final_test(test_dataset)
    #wandb.finish()
        
        
    
        
    


    
    









    

        