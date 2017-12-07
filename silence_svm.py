# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:23:56 2016

@author: maureen
"""

import numpy as np
from sklearn import svm
import logging
from speech_recognition.silence_detection import SilenceDetector
import glob
import os
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import mean_squared_error, explained_variance_score, \
#     r2_score
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from scipy.stats import pearsonr

logging.basicConfig(level=logging.DEBUG)


def extract_features(sd, fnames):
    features_all = []
    for fname in fnames:
        features = []
        signal = sd.sd_preprocess(fname, is_plot=False)
        a_corr = sd.autocorrelation(signal)
        features.append(a_corr)
        features_all.append(features)
    return features_all


if __name__ == "__main__":
    SD = SilenceDetector()
    silence_pathnames = 'assets/data_augmentation/silence/background/*.wav'
    all_train_pathnames = 'assets/train/audio/*/*wav'
    bn_dir = "assets/train/audio/_background_noise_/"
    silence_fnames = glob.glob(silence_pathnames)
    speech_fnames = [x for x in glob.glob(all_train_pathnames) if
                             os.path.dirname(x) is not bn_dir]

    Y_silence = np.empty(len(silence_fnames))
    Y_silence.fill(0)
    Y_speech = np.empty(len(speech_fnames))
    Y_speech.fill(1)
    Y = np.concatenate([Y_silence, Y_speech], axis=0)

    fnames = silence_fnames + speech_fnames
    data = [(fnames[i], Y[i]) for i in range(len(fnames))]
    np.random.shuffle(data)
    fnames = [x[0] for x in data]
    Y = [x[1] for x in data]
    X = extract_features(SD, fnames)

    # ss = preprocessing.StandardScaler()
    # mm = preprocessing.StandardScaler()
    # x_scale = ss.fit_transform(X)
    # y_scale = mm.fit_transform(Y)

    isCrossValidation = True
    isTest = False
    isPlot = False

    svc_rbf = svm.SVC(kernel='rbf')
    scores = cross_val_score(svc_rbf, X, Y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


    # seed = 7
    # np.random.seed(seed)
    #
    # if isCrossValidation:
    #     listOfAvgR2 = []
    #     j = 0
    #
    #     for n_output in range(len(y_scale[0, :])):
    #         kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    #         listOfMSE = []
    #         listOfR2 = []
    #         listOfScore = []
    #         listOfHistory = []
    #         i = 0
    #         print("n_output", j)
    #         for train_index, test_index in kfold.split(x_scale):
    #             print("kfold :", str(i + 1))
    #             history = svc_rbf.fit(x_scale[train_index],
    #                                   y_scale[train_index, n_output])
    #             output = history.predict(x_scale[test_index])
    #             target = y_scale[test_index, n_output]
    #             prediction = output
    #             r2Score = r2_score(target, prediction)
    #             listOfR2.append(r2Score)
    #             i += 1
    #         avg_r2Score = np.mean(listOfR2, axis=0)
    #         avg_score = np.mean(listOfScore)
    #         print("average R2:", avg_r2Score)
    #         listOfAvgR2.append(avg_r2Score)
    #         j += 1
    #     avgAvgR2 = np.mean(listOfAvgR2)
    #     print(avgAvgR2)