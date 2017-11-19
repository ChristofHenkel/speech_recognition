"""
# Good MFCC explanation: 
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""
import re
from glob import glob
import os
import numpy as np
from scipy.io import wavfile
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)


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


class SoundCorpusCreator:
    """Docu goes here"""
    def __init__(self, data, mode = 'train'):
        self.data = data
        self.mode = mode
        self.config = {}
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
                    # wav = self.preprocessing(wav)
                    """
                    # This might be helpful to compute mfcc
                    y, sr = librosa.load(librosa.util.example_audio_file())
                    S = librosa.feature.melspectrogram(y=y, sr=sr,n_mels=128, fmax=8000)
                    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))                    
                    """
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
    
    
    def preprocessing(self, wav):
        wav = self.preempahsis(wav)
        return wav
    
    @staticmethod
    def preemphasis(signal):
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        return emphasized_signal
    
    def build_corpus(self, max = None, fn = None):
        corpus = []
        k = 0
        for date in self:
            if k % 100 == 0:
                logging.debug('progress: ' + str(k))
            corpus.append(date)
            k += 1
        if fn is None:
            return corpus
        else:
            with open(fn,'wb') as f:
                pickle.dump(corpus,f)
            return 0


if __name__ == '__main__':
    DATADIR = 'assets/'
    SAVE_PATH = DATADIR + 'corpora/corpus1/'

    trainset, valset = load_data(DATADIR)
    gen_train = SoundCorpusCreator(trainset)

    train_corpus = gen_train.build_corpus()

    # using incrementally pickle to stream from later
    with open(SAVE_PATH + 'train.soundcorpus.p','wb') as f:
        pickler = pickle.Pickler(f)
        for e in train_corpus:
            pickler.dump(e)

    gen_val = SoundCorpusCreator(valset)

    val_corpus = gen_val.build_corpus()

    with open(SAVE_PATH + 'valid.soundcorpus.p', 'wb') as f:
        pickler = pickle.Pickler(f)
        for e in val_corpus:
            pickler.dump(e)

    # should also save len of train and valid data
    with open(SAVE_PATH + 'nameiddict.p','wb') as f:
        pickle.dump((id2name,name2id),f)
