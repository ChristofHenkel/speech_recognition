import re
from glob import glob
import os
import numpy as np
from scipy.io import wavfile
import logging
import random
import pickle
from python_speech_features import mfcc, logfbank, delta

logging.basicConfig(level=logging.DEBUG)

class SC_Config:
    def __init__(self, mode='train'):
        self.padding = True
        self.mfcc = False
        self.pure_silence_portion = 0.1 # How  much of the resulting data should be pure silence.
        self.background_silence_portion = 0 # How  much of the resulting data should be silence from background.
        self.unknown_portion = 0.1 # How much should be audio outside the wanted classes.
        self.possible_labels = 'yes no up down left right on off stop go silence unknown'.split()
        self.id2name = {i: name for i, name in enumerate(self.possible_labels)}
        self.name2id = {name: i for i, name in self.id2name.items()}
        self.mode = mode
        self.modes = ['train','valid','test','only_background','only_unknown']
        self.data_root = 'assets/'
        self.dir_files = 'train/audio/*/*wav'
        self.validation_list_fp = 'train/validation_list.txt'
        self.save_dir = self.data_root + 'corpora/corpus10/'
        self.seed = np.random.seed(1)
        self.paths_test = glob(os.path.join('assets', 'test/audio/*wav'))
        self.dir_background_noise = 'assets/data_augmentation/silence/background/'
        self.L = 16000 # length of files


class SoundCorpusCreator:
    """Docu goes here"""

    def __init__(self, config):

        self.config = config
        self.data = None
        self.flags = []
        if self.config.padding:
            self.flags.append('p')
        if self.config.mfcc:
            self.flags.append('m')
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        if self.config.mode in ['train','valid']:
            self.train_data, self.valid_data = self.load_data()
            # self.data = self.set_data()
          #  self.size = None
          #  self.set_data()
        elif self.config.mode == 'test':
            self.test_data = self.config.paths_test
        self.L = self.config.L

    def _update_config(self,cfg):
        self.config = cfg

    #def set_data(self):
    #    if self.config.mode == 'train':
    #        self.data = self.train_data
    #    elif self.config.mode == 'valid':
    #        self.data = self.valid_data
    #    else:
    #        logging.warning('wrong mode')
    #    self.size = len(self.data)

    def _do_mfcc(self,signal):
        signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                      nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                      ceplifter=22, appendEnergy=True)
        dsignal = delta(signal, N=1)
        ddsignal = delta(dsignal, N=1)
        signal = np.stack([signal, dsignal, ddsignal], axis=2)
        return signal

    def _read_wav_and_pad(self,fname):
        _, wav = wavfile.read(fname)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max

        # be aware, some files are shorter than 1 sec!
        # if label_id is unknow drop 90%
        len_wav = wav.shape[0]
        if len_wav < self.L:
            if self.config.padding:
                # todo: test
                # randomly insert wav into a 16k zero pad
                padded = np.zeros([self.L])
                start = np.random.randint(0, self.L - len_wav)
                end = start + len_wav
                padded[start:end] = wav
                wav = padded
        if len_wav > self.L:
            print(len_wav)
            beg = np.random.randint(0, len_wav - self.L)
        else:
            beg = 0

        signal = wav[beg: beg + self.L]
        return signal

    def data_gen(self, mode):
        if mode in ['train','valid']:
            if mode == 'train':
                data = self.train_data
                size = len(data)
                for _ in range(int(size * self.config.pure_silence_portion)):
                    data.append((self.config.name2id['silence'], '', 'assets/data_augmentation/silence/pure_silence.wav'))
                background_silence_fns = os.listdir(self.config.dir_background_noise)
                background_silence_fns = [fn for fn in background_silence_fns if fn.endswith('.wav')]
                for _ in range(int(size * self.config.background_silence_portion)):
                    fn = self.config.dir_background_noise + random.choice(background_silence_fns)
                    data.append((self.config.name2id['silence'], '',fn))

            elif mode == 'valid':
                data = self.valid_data
            elif mode == 'test':
                data = self.test_data
            else:
                data = None
                logging.critical('wrong mode')
            np.random.shuffle(data)
            # Feel free to add any augmentation
            for (label_id, uid, fname) in data:
                try:
                    signal = self._read_wav_and_pad(fname)



                    if self.config.background_silence_portion > 0 and label_id == self.config.name2id['silence']:
                        # make background noise more silent:
                        signal = signal / np.random.uniform(2,10)

                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)


                    yield dict(
                        target=np.int32(label_id),
                        wav=signal,
                    )

                except Exception as err:
                    print(err, label_id, uid, fname)
        elif self.config.mode in ['test']:
            for path in self.test_data:
                signal = self._read_wav_and_pad(path)
                fname = os.path.basename(path)
                if self.config.mfcc:
                    signal = self._do_mfcc(signal)
                yield dict(
                    sample=np.string_(fname),
                    wav=signal,
                )
        elif self.config.mode == 'only_background':

            data = []
            background_silence_fns = os.listdir(self.config.dir_background_noise)
            background_silence_fns = [self.config.dir_background_noise + fn for fn in background_silence_fns if fn.endswith('.wav')]
            for fn in background_silence_fns:
                data.append((99, '', fn))
            np.random.shuffle(data)
                # Feel free to add any augmentation
            for (label_id, uid, fname) in data:
                try:
                    signal = self._read_wav_and_pad(fname)

                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)

                    yield dict(
                        target=np.int32(label_id),
                        wav=signal,
                    )

                except Exception as err:
                    print(err, label_id, uid, fname)
    def load_data(self, ignore_background=True):
        """ Return 2 lists of tuples:
        [(class_id, user_id, path), ...] for train
        [(class_id, user_id, path), ...] for validation
        """
        # Just a simple regexp for paths with three groups:
        # prefix, label, user_id
        pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
        all_files = glob(os.path.join(self.config.data_root, self.config.dir_files))

        with open(os.path.join(self.config.data_root, self.config.validation_list_fp), 'r') as fin:
            validation_files = fin.readlines()
        valset = set()
        for entry in validation_files:
            r = re.match(pattern, entry)
            if r:
                valset.add(r.group(3))

        possible = set(self.config.possible_labels)
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

                label_id = self.config.name2id[label]

                sample = (label_id, uid, entry)
                if label == 'unknown':
                    if np.random.rand() > self.config.unknown_portion:
                        continue
                if not label == 'ignore':
                    if uid in valset:
                        val.append(sample)
                    else:
                        train.append(sample)

        print('There are {} train and {} val samples'.format(len(train), len(val)))
        return train, val


    def build_train_and_val_corpus(self):
        len_train = self.build_corpus('train')
        len_valid = self.build_corpus('valid')
        return len_train, len_valid

    def build_corpus(self, mode):
        corpus = []

        k = 0
        for date in self.data_gen(mode = mode):
            if k % 100 == 0:
                logging.debug('progress: ' + str(k) + '/' )
            corpus.append(date)
            k += 1
        save_name = self.config.save_dir
        save_name += mode + '.'
        save_name += ''.join(self.flags)
        save_name += '.soundcorpus.p'
        logging.info('saving under: ' + save_name)
        with open(save_name, 'wb') as f:
            pickler = pickle.Pickler(f)
            for e in corpus:
                pickler.dump(e)
        return len(corpus)

    def _get_label_distribution(self, data):
        labels = []
        for d in data:
            label = d[2].split('/')[3]
            if label not in self.config.possible_labels:
                label = 'unknown'
            labels.append(label)
        all_labels = list(set(labels))
        count_dict = {}
        for item in all_labels:
            count_dict[item] = labels.count(item)
        return count_dict

    @staticmethod
    def change_volume(wav,factor):
        wav = wav / factor
        return wav

if __name__ == '__main__':

    cfg_train = SC_Config(mode='train')
    train_corpus = SoundCorpusCreator(cfg_train)
    len_train, len_valid = train_corpus.build_train_and_val_corpus()

    cfg_test = SC_Config(mode='test')
    test_corpus = SoundCorpusCreator(cfg_test)
    len_test = test_corpus.build_corpus('test')

    # should also save len of corpora
    info_dict = {'id2name': cfg_train.id2name,
                 'name2id': cfg_train.name2id,
                 'len_train': len_train,
                 'len_valid': len_valid,
                 'len_test': len_test
                 }

    print(info_dict['id2name'])
    with open(cfg_train.save_dir + 'infos.p','wb') as f:
        pickle.dump(info_dict,f)

    cfg_bg = SC_Config(mode='only_background')
    train_corpus = SoundCorpusCreator(cfg_bg)
    len_bg = train_corpus.build_corpus('only_background')