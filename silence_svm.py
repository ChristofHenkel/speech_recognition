# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:23:56 2016

@author: maureen
"""

import numpy as np
import logging
import glob
import os
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
#import matplotlib.pyplot as plt
from silence_detection import SilenceDetector

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
        signal = sd.sd_preprocess(fname, is_plot=False)
        # a_corr = sd.autocorrelation(signal)
        # envelope = sd.get_amplitude_envelop(signal)
        mfcc = sd._read_wav_and_pad(fname)
        features_all.append(mfcc)
    return features_all


def get_corpus_subset(fnames, n):
    indicies = np.random.random_integers(0, len(fnames) - 1, n)
    fnames = np.asarray(fnames)
    fnames = fnames[indicies]
    return fnames


def get_balanced_corpus(silence_fnames, speech_fnames, num_of_data=-1,
                       percentage_of_speech=0.5,
                       test_size=0.33, random_state=42, is_split=False):
    SD = SilenceDetector()
    if num_of_data > 0:
        max_num_of_data = np.min([len(silence_fnames), len(speech_fnames)])
        if num_of_data > max_num_of_data:
            num_of_data = max_num_of_data
        print ("(Max) num of data: " + str(num_of_data))
        n_speech = int(percentage_of_speech * num_of_data)
        n_silence = int((1 - percentage_of_speech) * num_of_data)
        speech_fnames = get_corpus_subset(speech_fnames, n_speech)
        silence_fnames = get_corpus_subset(silence_fnames, n_silence)

    Y_silence = np.empty(len(silence_fnames))
    Y_silence.fill(0)
    Y_speech = np.empty(len(speech_fnames))
    Y_speech.fill(1)

    Y = np.concatenate([Y_silence, Y_speech], axis=0)

    if isinstance(speech_fnames, np.ndarray):
        fnames = silence_fnames.tolist() + speech_fnames.tolist()
    else:
        fnames = silence_fnames + speech_fnames
    data = [(fnames[i], Y[i]) for i in range(len(fnames))]
    np.random.shuffle(data)
    fnames = [x[0] for x in data]
    y = [x[1] for x in data]
    X = extract_features(SD, fnames)


    if is_split:
        ss = StandardScaler()
        x_scale = ss.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(x_scale, y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test, ss
    else:
        return X, y


if __name__ == "__main__":
    silence_pathnames = 'assets/data_augmentation/silence/background/*.wav'
    all_train_pathnames = 'assets/train/audio/*/*wav'
    bn_dir = "assets/train/audio/_background_noise_/"
    silence_fnames = glob.glob(silence_pathnames)
    speech_fnames = [x for x in glob.glob(all_train_pathnames) if
                     os.path.dirname(x) is not bn_dir]

    ##################### Load data set
    X_train, X_test, y_train, y_test, ss = get_balanced_corpus(silence_fnames,
                                                          speech_fnames, 8000,
                                                          0.5, is_split=True)



    ##################### Model
    svc_rbf = svm.SVC(kernel='rbf')


    ###################### Automatic Cross Validation
    cross_val = 2
    scores = cross_val_score(svc_rbf, X_train, y_train, cv=cross_val)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


###################### Training & Testing
    svc_rbf.fit(X_train, y_train)

    is_test1 = True
    is_test2 = True
    ####################### Test 1
    if is_test1:
        y_score = svc_rbf.decision_function(X_test)
        average_precision = average_precision_score(y_test, y_score)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        is_plot = False
        if is_plot:
            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2,
                             color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(
                '2-class Precision-Recall curve: AP={0:0.2f}'.format(
                    average_precision))
            plt.show()

    ####################### Test 2
    if is_test2:
        test_sil_pathnames = "../../kaggle_additional_data/label/silence/*.wav"
        test_all_pathnames = "../../kaggle_additional_data/label/*/*.wav"
        ts_sil_fnames = glob.glob(test_sil_pathnames)
        sil_dir = "../kaggle_additional_data/label/silence/"
        ts_no_sil_fnames = [x for x in glob.glob(test_all_pathnames) if
                            os.path.dirname(x) is not sil_dir]

        Xt, yt = get_balanced_corpus(ts_sil_fnames, ts_no_sil_fnames,
                                     is_split=False)
        Xt = ss.transform(Xt)
        output = svc_rbf.predict(Xt)
        y_score = svc_rbf.decision_function(Xt)
        average_precision = average_precision_score(yt, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))





#################### Old code
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
