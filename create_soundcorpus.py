import re
from glob import glob
import os
import numpy as np
from scipy.io import wavfile
import logging
import pickle
from python_speech_features import mfcc, logfbank, delta
from pydub import AudioSegment

logging.basicConfig(level=logging.DEBUG)

class SC_Config:
    def __init__(self, mode='train'):
        self.padding = True
        self.mfcc = True
        self.generate_silence = False
        self.add_silence = True #adding a silent file
        self.mode = mode
        self.modes = ['train','valid','test']
        self.data_dir = 'assets/'
        self.save_dir = self.data_dir + 'corpora/corpus7/'

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

def load_data(data_dir, ignore_background = True):
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
                if ignore_background:
                    label = 'ignore'
                else:
                    label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            if not label == 'ignore':
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
        if self.config.mode in ['train','valid']:
            if self.config.mode == 'train':
                if self.config.add_silence:
                    for _ in range(2000):
                        self.data.append((10, '', 'assets/test/audio/clip_0a1da4f17.wav'))
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
                        # continue

                    # let's generate more silence!
                    if self.config.generate_silence:
                        samples_per_file = 1 if label_id != name2id['silence'] else 20
                    else:
                        samples_per_file = 1
                    for _ in range(samples_per_file):
                        if len(wav) > L:
                            beg = np.random.randint(0, len(wav) - L)
                        else:
                            beg = 0

                        signal = wav[beg: beg + L]
                        if self.config.mfcc:
                            signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                                 nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                                 ceplifter=22, appendEnergy=True)
                            dsignal = delta(signal,N=1)
                            ddsignal = delta(dsignal, N=1)
                            signal = np.stack([signal,dsignal,ddsignal], axis = 2)


                        yield dict(
                            target=np.int32(label_id),
                            wav=signal,
                        )

                except Exception as err:
                    print(err, label_id, uid, fname)
        elif self.config.mode in ['test']:
            for path in self.data:
                _, wav = wavfile.read(path)
                signal = wav.astype(np.float32) / np.iinfo(np.int16).max
                L = 16000  # be aware, some files are shorter than 1 sec!

                len_wav = len(signal)
                if len_wav > L:
                    print(len_wav)
                if len_wav < L:
                    print(len_wav)
                    if self.config.padding:
                        # todo: test
                        # randomly insert wav into a 16k zero pad
                        padded = np.zeros([L])
                        start = np.random.randint(0, L - len_wav)
                        end = start + len_wav
                        padded[start:end] = wav
                        wav = padded
                fname = os.path.basename(path)
                if self.config.mfcc:
                    signal = mfcc(wav, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                                  nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                                  ceplifter=22, appendEnergy=True)
                    dsignal = delta(signal, N=1)
                    ddsignal = delta(dsignal, N=1)
                    signal = np.stack([signal, dsignal, ddsignal], axis=2)
                yield dict(
                    sample=np.string_(fname),
                    wav=signal,
                )

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
        return len(corpus)

    @staticmethod
    def change_volume(wav,factor):
        wav = wav / factor
        return wav

if __name__ == '__main__':

    cfg_train = SC_Config(mode='train')
    trainset, valset = load_data(cfg_train.data_dir)

    gen_train = SoundCorpusCreator(trainset,cfg_train)
    len_train = gen_train.build_corpus()

    cfg_valid = SC_Config(mode='valid')
    gen_val = SoundCorpusCreator(valset,cfg_valid)
    len_valid =  gen_val.build_corpus()

    paths = glob(os.path.join('assets', 'test/audio/*wav'))
    print(len(paths))
    cfg_test = SC_Config(mode='test')
    gen_test = SoundCorpusCreator(paths,cfg_test)
    len_test = gen_test.build_corpus()

    # should also save len of corpora
    info_dict = {'id2name': id2name,
                 'name2id': name2id,
                 'len_train': len_train,
                 'len_valid': len_valid,
                 'len_test': len_test
                 }

    print(info_dict['id2name'])
    with open(cfg_train.save_dir + 'infos.p','wb') as f:
        pickle.dump(info_dict,f)