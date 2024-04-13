import numpy as np
from scipy.io import wavfile
from scipy.special import logsumexp
from numpy import pi
from audioDetection2 import mel_inv, mel, mel_filter_bank, framing, spectrogram, mfcc, wav16khz2mfcc
import os


class GMM:
    def __init__(self, Ws_non_target, MUs_non_target, COVs_non_target, Ws_target, MUs_target, COVs_target, P_non_target, P_target, M_non_target, M_target):
        self.Ws_non_target = Ws_non_target
        self.MUs_non_target = MUs_non_target
        self.COVs_non_target = COVs_non_target
        self.P_non_target = P_non_target
        self.M_non_target = M_non_target

        self.Ws_target = Ws_target
        self.MUs_target = MUs_target
        self.COVs_target = COVs_target
        self.P_target = P_target
        self.M_target = M_target

    @staticmethod
    def logpdf_gauss(x, mu, cov):
        assert(mu.ndim == 1 and len(mu) == len(cov) and (cov.ndim == 1 or cov.shape[0] == cov.shape[1]))
        x = np.atleast_2d(x) - mu
        if cov.ndim == 1:
            return -0.5*(len(mu)*np.log(2 * np.pi) + np.sum(np.log(cov)) + np.sum((x**2)/cov, axis=1))
        else:
            return -0.5*(len(mu)*np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + np.sum(x.dot(np.linalg.inv(cov)) * x, axis=1))

    @staticmethod
    def logpdf_gmm(x, component_index):
        if component_index == 0:
            ws = gmm_model.Ws_non_target
            mus = gmm_model.MUs_non_target
            covs = gmm_model.COVs_non_target
        else:
            ws = gmm_model.Ws_target
            mus = gmm_model.MUs_target
            covs = gmm_model.COVs_target
        return logsumexp([np.log(w) + GMM.logpdf_gauss(x, m, c) for w, m, c in zip(ws, mus, covs)], axis=0)
    
    @staticmethod
    def loadGMM(filename):
        with np.load(filename) as data:
            Ws_non_target = data['Ws_non_target']
            MUs_non_target = data['MUs_non_target']
            COVs_non_target = data['COVs_non_target']
            M_non_target = data['M_non_target']
            P_non_target = data['P_non_target']

            Ws_target = data['Ws_target']
            MUs_target = data['MUs_target']
            COVs_target = data['COVs_target']
            M_target = data['M_target']
            P_target = data['P_target']
        return GMM(Ws_non_target, MUs_non_target, COVs_non_target, Ws_target, MUs_target, COVs_target, P_non_target, P_target, M_non_target, M_target)




if __name__ == '__main__':
    # GMM model
    gmm_model = GMM.loadGMM('audioModelGMM.npz')

    dataPath = os.getcwd() + "/data/dev"
    target_dev = list(wav16khz2mfcc(os.path.join(dataPath,"target_dev")).values())

    score=[] #True = target, False = non_target
    for tst in target_dev:
        ll_non_target = gmm_model.logpdf_gmm(tst, 0)
        ll_target = gmm_model.logpdf_gmm(tst, 1)
        score.append((sum(ll_non_target) + np.log(gmm_model.P_non_target)) - (sum(ll_target) + np.log(gmm_model.P_target)) < 0) 

    print(score)