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

def logpdf_gauss(x, mu, cov):
    assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
    x = np.atleast_2d(x) - mu
    if cov.ndim == 1:
        return -0.5*(len(mu)*np.log(2 * pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
    else:
        return -0.5*(len(mu)*np.log(2 * pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(np.linalg.inv(cov)) * x, axis=1))


def train_gauss(x):
    """
    Estimates gaussian distribution from data.
    (MU, COV) = TRAIN_GAUSS(X) return Maximum Likelihood estimates of mean
    vector and covariance matrix estimated using training data X
    """
    return np.mean(x, axis=0), np.cov(x.T, bias=True)


def train_gmm(x, ws, mus, covs):
    """
    TRAIN_GMM Single iteration of EM algorithm for training Gaussian Mixture Model
    [Ws_new,MUs_new, COVs_new, TLL]= TRAIN_GMM(X,Ws,NUs,COVs) performs single
    iteration of EM algorithm (Maximum Likelihood estimation of GMM parameters)
    using training data X and current model parameters Ws, MUs, COVs and returns
    updated model parameters Ws_new, MUs_new, COVs_new and total log likelihood
    TLL evaluated using the current (old) model parameters. The model
    parameters are mixture component mean vectors given by columns of M-by-D
    matrix MUs, covariance matrices given by M-by-D-by-D matrix COVs and vector
    of weights Ws.
    """   
    gamma = np.vstack([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)])
    logevidence = logsumexp(gamma, axis=0)
    gamma = np.exp(gamma - logevidence)
    tll = logevidence.sum()
    gammasum = gamma.sum(axis=1)
    ws = gammasum / len(x)
    mus = gamma.dot(x)/gammasum[:,np.newaxis]
    
    if covs[0].ndim == 1: # diagonal covariance matrices
      covs = gamma.dot(x**2)/gammasum[:,np.newaxis] - mus**2
    else:
      covs = np.array([(gamma[i]*x.T).dot(x)/gammasum[i] - mus[i][:, newaxis].dot(mus[[i]]) for i in range(len(ws))])

    covs = check_minimum_cov(covs)
    return ws, mus, covs, tll


def logpdf_gmm(x, ws, mus, covs):
    return logsumexp([np.log(w) + logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)


def check_minimum_cov(covs, epsilon=1e-6, fraction=0.1):
    """Check if any covariance matrix is overfitted on a single data point"""
    avg_diagonal = np.mean([np.mean(np.diag(c)) for c in covs])  # Calculate average diagonal element across all covariances
    threshold = fraction * avg_diagonal  # Define threshold as a fraction of the average diagonal element
    
    for i in range(len(covs)):
        # Check if any diagonal element is significantly smaller than the threshold
        if covs[i].ndim == 1:
            min_diagonal = np.min(covs[i])
        else:
            min_diagonal = np.min(np.diag(covs[i]))
        if min_diagonal < threshold:
            # Add a small positive constant to the diagonal elements of the covariance matrix
            covs[i] += epsilon * np.eye(covs[i].shape[0])
    return covs


if __name__ == '__main__':
    dataPath = os.getcwd() + "/data"
    dirs = ["non_target_train", "target_train","non_target_dev", "target_dev"] 

    non_target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[0])).values())
    target_train = list(wav16khz2mfcc(os.path.join(dataPath,dirs[1])).values())
    non_target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[2])).values())
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,dirs[3])).values())

    non_target_train = np.vstack(non_target_train)
    target_train = np.vstack(target_train)

    dim = non_target_train.shape[1]

    cov_tot = np.cov(np.vstack([non_target_train, target_train]).T, bias=True)
    # take just 2 largest eigenvalues and corresponding eigenvectors
    d, e = scipy.linalg.eigh(cov_tot, subset_by_index=(dim-2, dim-1))

    non_target_train_pca = non_target_train.dot(e)
    target_train_pca = target_train.dot(e)
    """"
    plt.plot(non_target_train_pca[:,1], non_target_train_pca[:,0], 'b.', ms=1)
    plt.plot(target_train_pca[:,1], target_train_pca[:,0], 'r.', ms=1)
    plt.show()
    """
    
    # LDA reduction to 1 dimenzion (only one LDA dimension is available for 2 tridy)
    n_m = len(non_target_train)
    n_f = len(target_train)
    cov_wc = (n_m*np.cov(non_target_train.T, bias=True) + n_f*np.cov(target_train.T, bias=True)) / (n_m + n_f)
    cov_ac = cov_tot - cov_wc
    d, e = scipy.linalg.eigh(cov_ac, cov_wc, subset_by_index=(dim-1, dim-1))

    """
    plt.figure()
    junk = plt.hist(non_target_train.dot(e), 40, histtype='step', color='b', density=True)
    junk = plt.hist(target_train.dot(e), 40, histtype='step', color='r', density=True)
    plt.show()
    """
    # Lets define uniform a-priori probabilities of classes:
    P_m = 0.5
    P_f = 1 - P_m   

    # Train and test with GMM models with diagonal covariance matrices
    # Decide for number of gaussian mixture components used for the male model
    M_m = 5

    # Initialize mean vectors, covariance matrices and weights of mixture componments
    # Initialize mean vectors to randomly selected data points from corresponding class
    MUs_m  = non_target_train[randint(1, len(non_target_train), M_m)]

    # Initialize all variance vectors (diagonals of the full covariance matrices) to
    # the same variance vector computed using all the data from the given class
    COVs_m = [np.var(non_target_train, axis=0)] * M_m

    # Use uniform distribution as initial guess for the weights
    Ws_m   = np.ones(M_m) / M_m;


    # Initialize parameters of feamele model
    M_f = 3
    MUs_f  = target_train[randint(1, len(target_train), M_f)]
    COVs_f = [np.var(target_train, axis=0)] * M_f
    Ws_f   = np.ones(M_f) / M_f;

    jj = 0 
    TTL_m_old = 0
    TTL_f_old = 0

    # Run 30 iterations of EM algorithm to train the two GMMs from males and females
    
    while True:
        [Ws_m, MUs_m, COVs_m, TTL_m] = train_gmm(non_target_train, Ws_m, MUs_m, COVs_m); 
        [Ws_f, MUs_f, COVs_f, TTL_f] = train_gmm(target_train, Ws_f, MUs_f, COVs_f); 
        if abs(TTL_m - TTL_m_old) < 1 and abs(TTL_f - TTL_f_old) < 1:
            break
        print('Iteration:', jj, ' Total log-likelihood:', TTL_m, 'for non_target;', TTL_f, 'for target')
        TTL_m_old = TTL_m
        TTL_f_old = TTL_f
        jj += 1

    # To do the same for females set "test_set=test_f"
    score=[]
    for tst in non_target_dev:
        ll_m = logpdf_gmm(tst, Ws_m, MUs_m, COVs_m)
        ll_f = logpdf_gmm(tst, Ws_f, MUs_f, COVs_f)
        score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)) > 0)
    print(score)

    score=[]
    for tst in target_dev:
        ll_m = logpdf_gmm(tst, Ws_m, MUs_m, COVs_m)
        ll_f = logpdf_gmm(tst, Ws_f, MUs_f, COVs_f)
        score.append((sum(ll_m) + np.log(P_m)) - (sum(ll_f) + np.log(P_f)) < 0)
    print(score)
    




