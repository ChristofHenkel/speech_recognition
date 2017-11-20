import re
from glob import glob
import os
import numpy as np
from scipy.io import wavfile
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)

class SC_Config:
    def __init__(self, mode='train'):
        self.padding = True
        self.mfcc = False
        self.mode = mode
        self.data_dir = 'assets/'
        self.save_dir = self.data_dir + 'corpora/corpus3/'

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
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.size = len(self.data)
        self.flags = []
        if self.config.padding:
            self.flags.append('p')
        if self.config.mfcc:
            self.flags.append('m')
    def __iter__(self):
        if self.config.mode == 'train':
            np.random.shuffle(self.data)
        # Feel free to add any augmentation
        for (label_id, uid, fname) in self.data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!

                len_wav = len(wav)
                if len_wav < L:
                    if self.config.padding:
                        # todo: test
                        # randomly insert wav into a 16k zero pad
                        padded = np.zeros([L])
                        start = np.random.randint(0,L-len_wav)
                        end = start + len_wav
                        padded[start:end] = wav
                        wav = padded

                    # wav = self.preprocessing(wav)
                    """
                    # This might be helpful to compute mfcc
                    y, sr = librosa.load(librosa.util.example_audio_file())
                    S = librosa.feature.melspectrogram(y=y, sr=sr,n_mels=128, fmax=8000)
                    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S))                    
                    """
                    # continue

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
        wav = self.preemphasis(wav)
        return wav
    

    def preemphasis(self, wav):
        pre_emphasis = 0.97
        emphasized_signal = np.append(wav[0], wav[1:] - pre_emphasis * wav[:-1])
        return emphasized_signal
    
    def build_corpus(self):
        corpus = []
        k = 0
        for date in self:
            if k % 100 == 0:
                logging.debug('progress: ' + str(k) + '/' + str(self.size))
            corpus.append(date)
            k += 1
        save_name = self.config.save_dir
        save_name += self.config.mode + '.'
        save_name += ''.join(self.flags)
        save_name += '.soundcorpus.p'
        logging.info('saving under: ' + save_name)
        with open(save_name, 'wb') as f:
            pickler = pickle.Pickler(f)
            for e in corpus:
                pickler.dump(e)


if __name__ == '__main__':

    cfg_train = SC_Config(mode='train')
    trainset, valset = load_data(cfg_train.data_dir)


    gen_train = SoundCorpusCreator(trainset,cfg_train)
    gen_train.build_corpus()

    # using incrementally pickle to stream from later

    cfg_valid = SC_Config(mode='valid')
    gen_val = SoundCorpusCreator(valset,cfg_valid)
    gen_val.build_corpus()

    # should also save len of train and valid data
    info_dict = {'id2name': id2name,
                 'name2id': name2id,
                 'len_train': len(trainset),
                 'len_valid': len(valset)
                 }
    with open(cfg_train.save_dir + 'infos.p','wb') as f:
        pickle.dump(info_dict,f)
