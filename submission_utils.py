import os
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix


csv_file = 'assets/corpora/corpus3/t3_model17_e29_2_probs.csv'
csv_file2 = 'assets/corpora/corpus3/tmp_model14_2_probs.csv'
csv_file3 = 'assets/corpora/corpus3/t3_model16_e47_2_probs.csv'
csv_file4 = 'assets/corpora/corpus3/vgg_2_probs.csv'
csv_file5 = 'assets/corpora/corpus4/vgg_3_probs.csv'
csv_file6 = 'assets/corpora/corpus4/vgg_4_e17_probs.csv'
csv_file_silence = 'assets/corpora/corpus3/silence_detection_e1_s105_probs.csv'


def load_submission(csv_file):
    with open(csv_file, 'r') as f:
        content = [line.strip() for line in f.readlines()]

    submission= dict()
    for l in content[1:]:
        fname, label = l.split(',')
        submission[fname] = label
    return submission

def load_submission_probs(csv_file):
    with open(csv_file, 'r') as f:
        content = [line.strip() for line in f.readlines()]

    submission= dict()
    submission_probs = dict()
    for l in content[1:]:
        fname, label = l.split(',')[:2]
        prob = l.split(',')[2:]
        submission[fname] = label
        submission_probs[fname] = [float(p) for p in prob]
    return submission, submission_probs

def acc_on_own_test(submission):
        labels = set(fname2label.values())
        for l in labels:
            fns = [fn for fn in fname2label if fname2label[fn] == l]
            correct_l = [1 for fn in fns if decoder[fname2label[fn]] == submission[fn]]
            acc_l = sum(correct_l) / len(fns)
            print(decoder[l] + ' %s' %acc_l)
        correct = [1 for fn in fname2label if decoder[fname2label[fn]] == submission[fn]]
        accuracy = sum(correct) / len(fname2label)
        print(accuracy)

def combine_submission_probs(submission_list,submission_probs_list):
    new_submission = {}
    indices = {}
    for k in range(len(submission_list)):
        indices[k] = 0
    for i,fn in enumerate(submission_list[0]):
        if i % 1000 == 0:
            print(i)
        max_val = 0
        max_ind = 0
        for k in range(len(submission_list)):
            if submission_probs_list[k][fn][0] > max_val:
                max_val = submission_probs_list[k][fn][0]
                max_ind = k
        new_submission[fn] = submission_list[max_ind][fn]
        indices[max_ind] +=1
    print(indices)
    return new_submission

def combine_submission_probs2(submission_list,submission_probs_list):
    new_submission = {}
    new_submission_probs = {}
    probs = {}
    for i,fn in enumerate(submission_list[0]):
        if i % 1000 == 0:
            print(i)
        new_submission_probs[fn] = np.sum([np.power(sp[fn],0.5) for sp in submission_probs_list],axis = 0)
        new_submission[fn] = decoder[np.argmax(new_submission_probs[fn])]
        probs[fn] = np.max(new_submission_probs[fn])
    print(np.mean([probs[fn] for fn in probs]))
    return new_submission, new_submission_probs


def convert_silence_submission(submission_silence, submission_probs_silence):
    new_probs = {}
    for fn in submission_probs_silence:
        new_probs[fn] = [(1-submission_probs_silence[fn][0])/11 for i in range(11)] + [submission_probs_silence[fn][0]]
    return new_probs

def combine_submission_silence(submission, submission_silence, submission_probs):
    new_submission = {}
    for i, fn in enumerate(submission):
        if i % 100 == 0:
            print(i)
        if submission_silence[fn] == 'no':
            new_submission[fn] = 'silence'
        else:
            if submission[fn] == 'silence':
                new_submission[fn] = decoder[np.argsort(submission_probs[fn])[-2]]
            else:
                new_submission[fn] = submission[fn]
    return new_submission

def combine_submission_silence2(submission, submission_probs , submission_silence):
    new_submission = {}
    k1 = 0
    k2 = 0
    for i, fn in enumerate(submission):
        if i % 100 == 0:
            print(i)
        if submission_silence[fn] == 'no':
            if submission[fn] != 'silence':
                new_submission[fn] = 'silence'
                print(fn + ' ' + submission[fn] + ' -> ' + new_submission[fn])
                k1+=1
            else:
                new_submission[fn] = submission[fn]
        else:
            if submission[fn] == 'silence':
                new_submission[fn] = decoder[np.argsort(submission_probs[fn])[-2]]
                print(fn + ' ' + submission[fn] + ' -> ' + new_submission[fn])
                k2+=1
            else:
                new_submission[fn] = submission[fn]
    print('k1 %s' %k1)
    print('k2 %s' % k2)
    return new_submission

def combine_submission_silence3(submission, submission_probs , submission_silence, submission_silence_probs, min_diff = 0):
    new_submission = {}
    k1 = 0
    k2 = 0
    k3 = 0
    for i, fn in enumerate(submission):
        if i % 100 == 0:
            print(i)
        if submission_silence[fn] == 'no':
            if submission[fn] != 'silence':
                k3 +=1
                if 1-submission_silence_probs[fn][0]> np.max(submission_probs[fn]) + min_diff:
                    print('%s %s' %(1-submission_silence_probs[fn][0],np.max(submission_probs[fn])))
                    new_submission[fn] = 'silence'
                    print(fn + ' ' + submission[fn] + ' -> ' + new_submission[fn])
                    k1+=1
                else:
                    new_submission[fn] = submission[fn]
            else:
                new_submission[fn] = submission[fn]
        else:
            if submission[fn] == 'silence':
                k3 +=1
                if submission_silence_probs[fn][0] > np.max(submission_probs[fn]) + min_diff:
                    print('%s %s' % (submission_silence_probs[fn][0], np.max(submission_probs[fn])))
                    new_submission[fn] = decoder[np.argsort(submission_probs[fn])[-2]]
                    print(fn + ' ' + submission[fn] + ' -> ' + new_submission[fn])
                    k2+=1
                else:
                    new_submission[fn] = submission[fn]
            else:
                new_submission[fn] = submission[fn]
    print('k1 %s' %k1)
    print('k2 %s' % k2)
    print('k3 %s' % k3)
    return new_submission

def get_cm(submission):
    y_true = [fname2label[fn] for fn in submission]
    y_pred = [encoder[submission[fn]] for fn in submission]
    cm = confusion_matrix(y_true,y_pred)
    return cm

def get_quality(cm):
    quality = {}
    for r in range(cm.shape[0]):
        q = (2* cm[r,r]) / (sum(cm[r,:]) + sum(cm[:,r]))
        quality[decoder[r]] = q
    return quality

def write_submission(submission, fn_out):
    with open(fn_out, 'w') as fout:
        fout.write('fname,label\n')
        for fname, label in submission.items():
            fout.write('{},{}\n'.format(fname, label))

def acc(submission, print_cm = True):
    with open('assets/corpora/corpus3/' + 'fname2label.p', 'rb') as f:
        fname2label = pickle.load(f)
    with open('assets/corpora/corpus3/infos.p','rb') as f:
        infos = pickle.load(f)

    decoder = infos['id2name']
    encoder = infos['name2id']
    comparison = [(p,submission[p],decoder[fname2label[p]]) for p in submission]
    acc = [a[1] == a[2] for a in comparison].count(True)/len(comparison)
    no_silence = [c for c in comparison if c[2] != 'silence']
    acc_dict = {}
    for l in encoder:
        label_part = [c for c in comparison if c[2] == l]
        acc_label = [a[1] == a[2] for a in label_part].count(True)/len(label_part)
        acc_dict[l] = acc_label
    print(acc_dict)

    acc_no_silence = [a[1] == a[2] for a in no_silence].count(True)/len(no_silence)
    print('acc: %s' %acc)
    print('acc w/o silence: %s' % acc_no_silence)


def clear_other_labels(submission, label):
    cleared_submission = dict()
    for item in submission:
        if submission[item] != label:
            cleared_submission[item] = ' '
        else:
            cleared_submission[item] = submission[item]
    return cleared_submission

def all_one_labels(submission, label):
    cleared_submission = dict()
    for item in submission:
        if item != label:
            cleared_submission[item] = label
        else:
            cleared_submission[item] = submission[item]
    return cleared_submission


with open('assets/corpora/corpus3/' + 'fname2label.p', 'rb') as f:
    fname2label = pickle.load(f)
with open('assets/corpora/corpus3/infos.p','rb') as f:
    infos = pickle.load(f)

decoder = infos['id2name']
encoder = infos['name2id']

own_test = load_submission(csv_file2)


labels_true = [fname2label[item] for item in fname2label]
labels_predicted = [encoder[own_test[item]] for item in own_test]

def get_label_dist(labels):
    label_dist = {}
    for label in list(set(labels)):
        label_dist[label] = labels.count(label)/len(labels)
    return label_dist

label_dist_true = get_label_dist(labels_true)
label_dist_pred = get_label_dist(labels_predicted)

diff = {}
for item in list(set(labels_true)):
    diff[item] = label_dist_pred[item] - label_dist_true[item]

import operator

sorted_diff = sorted(diff.items(), key=operator.itemgetter(1), reverse=False)



##
_, probs = load_submission_probs(csv_file3_probs)

new_submission = {}
for l in [item[0] for item in sorted_diff]:

    k_max = int(label_dist_true[l]*len(labels_true))

    probs_l = {}
    for item in probs:
        probs_l[item] = float(probs[item][l])
    sorted_probs_l = sorted(probs_l.items(), key=operator.itemgetter(1), reverse=True)
    for t in sorted_probs_l[:k_max+1]:
        new_submission[t[0]] = decoder[l]

#classify missing as argmax
for item in own_test:
    if item not in new_submission:
        p = [float(i) for i in probs[item]]
        new_submission[item] = decoder[np.argmax(p)]

