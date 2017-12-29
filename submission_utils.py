import os

csv_file = 'assets/corpora/corpus14/tmp_model5.csv'

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
        fname, label, prob = l.split(',')
        submission[fname] = label
        submission_probs[fname] = prob
    return submission, submission_probs

