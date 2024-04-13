import numpy as np
from scipy.io import wavfile
from scipy.special import logsumexp
from numpy import pi
from audioDetectionGMM import mel_inv, mel, mel_filter_bank, framing, spectrogram, mfcc, wav16khz2mfcc, GMM
from audioDetection import MLP
import os


def _positive_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x):
    # Cache exp so you won't have to calculate it twice
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x):
    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=np.float)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


if __name__ == '__main__':

    # GMM model
    gmm_model = GMM.loadGMM('./trainedModels/audioModelGMM.npz')
    dataPath = os.getcwd() + "/data/dev"
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,"target_dev")).values())

    score=[] #True = target, False = non_target
    for tst in target_dev:
        ll_non_target = gmm_model.logpdf_gmm(tst, 0)
        ll_target = gmm_model.logpdf_gmm(tst, 1)
        score.append((sum(ll_non_target) + np.log(gmm_model.P_non_target)) - (sum(ll_target) + np.log(gmm_model.P_target)))

    print(score)