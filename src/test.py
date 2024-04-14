import numpy as np
from scipy.io import wavfile
from scipy.special import logsumexp
from numpy import pi
from audioDetectionGMM import mel_inv, mel, mel_filter_bank, framing, spectrogram, mfcc, wav16khz2mfcc, GMM, logistic_sigmoid
from visualDetection import load_model, predict
import torch
import os
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms



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
    
    # VGG model
    vgg_path = os.path.join(os.getcwd(), './trainedModels/vgg_best_model_fold_3.pth')
    vgg_model = load_model(vgg_path)
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4809, 0.3754, 0.3821], std=[0.2464, 0.2363, 0.2320])
    ])
    
    # Load the dev images dataset (target and non-target)
    dev_images = datasets.ImageFolder(os.path.join(os.getcwd(), './data/dev'), transform=transform)
    
    # Compute the scores (audio)
    score=[] #True = target, False = non_target
    for tst in target_dev:
        ll_non_target = gmm_model.logpdf_gmm(tst, 0)
        ll_target = gmm_model.logpdf_gmm(tst, 1)
        score.append(logistic_sigmoid(sum(ll_non_target) + np.log(gmm_model.P_non_target) - sum(ll_target) - np.log(gmm_model.P_target)) <= 0.5)

    for tst in non_target_dev:
        ll_non_target = gmm_model.logpdf_gmm(tst, 0)
        ll_target = gmm_model.logpdf_gmm(tst, 1)
        score.append(logistic_sigmoid(sum(ll_non_target) + np.log(gmm_model.P_non_target) - sum(ll_target) - np.log(gmm_model.P_target)) > 0.5)

    print(f"Audio score: {sum(score)/len(score)}")
    
    # Compute the scores (image)
    image_score = predict(vgg_model, image_path, transform)
    print(f"Image score: {image_score}")