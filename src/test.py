import numpy as np
from scipy.io import wavfile
from scipy.special import logsumexp
from numpy import pi
from audioDetectionGMM import mel_inv, mel, mel_filter_bank, framing, spectrogram, mfcc, wav16khz2mfcc, GMM, logistic_sigmoid
import torch
import os


def compute_score(audioScore, imageScore, threshold=0.5, fileName=None):
    # Compute the final score
    Wa = 1/2    # weight audio
    Wi = 1/2    # weight image
    if audioScore > 0.8:
        Wa = 3/4
        Wi = 1/4
    elif imageScore > 0.8:
        Wa = 1/4
        Wi = 3/4
    if Wa * audioScore + Wi * imageScore > threshold:
        print("Target")
    else:
        print("Non-target")



if __name__ == '__main__':

    
    # GMM model
    gmm_model = GMM.loadGMM('./trainedModels/audioModelGMM.npz')
    dataPath = os.getcwd() + "/data/dev"
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,"target_dev")).values())
    non_target_dev = list(wav16khz2mfcc(os.path.join(dataPath,"non_target_dev")).values())
    
    score=[] #True = target, False = non_target
    for tst in target_dev:
        ll_non_target = gmm_model.logpdf_gmm(tst, 0)
        ll_target = gmm_model.logpdf_gmm(tst, 1)
        score.append(logistic_sigmoid(sum(ll_non_target) + np.log(gmm_model.P_non_target) - sum(ll_target) - np.log(gmm_model.P_target)) <= 0.5)

    for tst in non_target_dev:
        ll_non_target = gmm_model.logpdf_gmm(tst, 0)
        ll_target = gmm_model.logpdf_gmm(tst, 1)
        score.append(logistic_sigmoid(sum(ll_non_target) + np.log(gmm_model.P_non_target) - sum(ll_target) - np.log(gmm_model.P_target)) > 0.5)

    print(sum(score)/len(score))
    
