import re
from glob import glob
import os
import numpy as np
from scipy.io import wavfile


DATADIR = 'assets/'
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

def data_generator(data, params, mode='train'):
    def generator():
        if mode == 'train':
            np.random.shuffle(data)
        # Feel free to add any augmentation
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    yield dict(
                        target=np.int32(label_id),
                        wav=wav[beg: beg + L],
                    )
            except Exception as err:
                print(err, label_id, uid, fname)

    return generator

class SoundCorpusFileStreamer:
    """Docu goes here"""
    def __init__(self, data, mode = 'train'):
        self.data = data
        self.mode = mode
    def __iter__(self):
        if self.mode == 'train':
            np.random.shuffle(self.data)
        # Feel free to add any augmentation
        for (label_id, uid, fname) in self.data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        beg = np.random.randint(0, len(wav) - L)
                    else:
                        beg = 0
                    yield dict(
                        target=np.int32(label_id),
                        wav=wav[beg: beg + L],
                    )
            except Exception as err:
                print(err, label_id, uid, fname)
    def build_corpus(self, max = None):
        corpus = []
        k = 0
        for date in self:
            if k < 100:
                corpus.append(date)
            k += 1
        return corpus



trainset, valset = load_data(DATADIR)

gen = SoundCorpusFileStreamer(trainset)

c = gen.build_corpus()
