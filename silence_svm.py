# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:23:56 2016

@author: maureen
"""

import numpy as np
import logging
import glob
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from speech_recognition.silence_detection import SilenceDetector

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
    n = len(fnames)
    for i, fname in enumerate(fnames):
        logging.log(logging.DEBUG, str(i) + "/" + str(n))
        features = []
        signal = sd.sd_preprocess(fname, is_plot=False)
        a_corr = sd.autocorrelation(signal)
        envelope = sd.get_amplitude_envelop(signal)
        mfcc = sd.read_wav_and_pad(fname)
        #features.append(a_corr)
        features_all.append(envelope)
    return features_all


if __name__ == "__main__":
    SD = SilenceDetector()
    silence_pathnames = 'assets/data_augmentation/silence/background/*.wav'
    all_train_pathnames = 'assets/train/audio/*/*wav'
    bn_dir = "assets/train/audio/_background_noise_/"
    silence_fnames = glob.glob(silence_pathnames)[:2000]
    speech_fnames = [x for x in glob.glob(all_train_pathnames) if
                             os.path.dirname(x) is not bn_dir][:2000]
    test_sil_pathnames = "../../kaggle_additional_data/label/silence/*.wav"
    test_all_pathnames = "../../kaggle_additional_data/label/*/*.wav"
    ts_sil_fnames = glob.glob(test_sil_pathnames)
    sil_dir = "../kaggle_additional_data/label/silence/"
    ts_no_sil_fnames = [x for x in glob.glob(test_all_pathnames) if
                        os.path.dirname(x) is not sil_dir]

    Y_silence = np.empty(len(silence_fnames))
    Y_silence.fill(0)
    Y_speech = np.empty(len(speech_fnames))
    Y_speech.fill(1)

    Y = np.concatenate([Y_silence, Y_speech], axis=0)

    fnames = silence_fnames + speech_fnames
    data = [(fnames[i], Y[i]) for i in range(len(fnames))]
    np.random.shuffle(data)
    fnames = [x[0] for x in data]
    y = [x[1] for x in data]
    X = extract_features(SD, fnames)

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y, test_size=0.33, random_state=42)

    svc_rbf = svm.SVC(kernel='rbf')
    # scores = cross_val_score(svc_rbf, X_train, y_train, cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    svc_rbf.fit(X_train, y_train)

    Yt_silence = np.empty(len(ts_sil_fnames))
    Yt_silence.fill(0)
    Yt_speech = np.empty(len(ts_no_sil_fnames))
    Yt_speech.fill(1)

    yt = np.concatenate([Yt_silence, Yt_speech], axis=0)
    tfnames = ts_sil_fnames + ts_no_sil_fnames
    Xt = extract_features(SD, tfnames)
    output = svc_rbf.predict(Xt)
    # for i, ot in enumerate(output):
    #     print(ot, yt[i])

    y_score = svc_rbf.decision_function(Xt)
    average_precision = average_precision_score(yt, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    # y_score = svc_rbf.decision_function(X_test)
    # print(y_score)
    # average_precision = average_precision_score(y_test, y_score)
    #
    # print('Average precision-recall score: {0:0.2f}'.format(
    #     average_precision))
    #
    # precision, recall, _ = precision_recall_curve(y_test, y_score)
    #
    # plt.step(recall, precision, color='b', alpha=0.2,
    #          where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2,
    #                  color='b')
    #
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #     average_precision))
    # plt.show()
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