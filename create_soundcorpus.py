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
        self.amount_silence = 4000 #number of training data with label silence
        self.pure_silence_portion = 0 # How  much silence should be pure silence the rest is sampled from backgroundnoise.
        self.background_silence_portion = 0 # How  much of the resulting data should be silence from background.
        self.unknown_portion = 0 # How much should be audio outside the wanted classes.
        self.possible_labels = 'yes no up down left right on off stop go unknown silence'.split()
        self.id2name = {i: name for i, name in enumerate(self.possible_labels)}
        self.name2id = {name: i for i, name in self.id2name.items()}
        self.mode = mode
        assert self.mode in ['train','valid','test','background','unknown','silence']
        self.data_root = 'assets/'
        self.dir_files = 'train/audio/*/*wav'
        self.validation_list_fp = 'train/validation_list.txt'
        self.save_dir = self.data_root + 'corpora/corpus12/'
        self.seed = np.random.seed(1)
        self.paths_test = glob(os.path.join('assets', 'test/audio/*wav'))
        self.dir_noise = 'assets/data_augmentation/silence/background/'
        self.L = 16000 # length of files
        self.noise2noise = True


class SoundCorpusCreator:
    """Docu goes here"""

    def __init__(self, config):

        self.config = config
        self.data = None
        self.name_flags = []
        if self.config.padding:
            self.name_flags.append('p')
        if self.config.mfcc:
            self.name_flags.append('m')
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        if self.config.mode in ['train','valid','unknown']:
            self.train_data, self.valid_data = self.load_data()
        elif self.config.mode == 'test':
            self.test_data = self.config.paths_test
        self.L = self.config.L

    @staticmethod
    def _do_mfcc(signal):
        signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                      nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                      ceplifter=22, appendEnergy=True)
        dsignal = delta(signal, N=1)
        ddsignal = delta(dsignal, N=1)
        signal = np.stack([signal, dsignal, ddsignal], axis=2)
        return signal

    def _read_wav_and_pad(self,fname):
        _, wav = wavfile.read(fname)
        # wav = wav.astype(np.float32) / np.iinfo(np.int16).max



        len_wav = wav.shape[0]
        if len_wav < self.L:    # be aware, some files are shorter than 1 sec!
            if self.config.padding:
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

    def data_gen(self):
        if self.config.mode == 'train':

            data = self.train_data
            data = [d for d in data if d[0] != self.config.name2id['unknown']]
            np.random.shuffle(data)
            for (label_id, uid, fname) in data:
                try:
                    signal = self._read_wav_and_pad(fname)

                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)

                    yield dict(label=np.int32(label_id),wav=signal,)

                except Exception as err:
                    print(err, label_id, uid, fname)
        elif self.config.mode in ['valid']:

            data = self.valid_data
            data_known = [d for d in data if d[0] != self.config.name2id['unknown']]
            data_unknown = [d for d in data if d[0] == self.config.name2id['unknown']]
            np.random.shuffle(data_unknown)
            end = int(len(data_known) * self.config.unknown_portion / (1-self.config.unknown_portion))
            data_unkown2 = data_unknown[:end]
            data = data_known
            data.extend(data_unkown2)
            np.random.shuffle(data)
            for (label_id, uid, fname) in data:
                try:
                    signal = self._read_wav_and_pad(fname)

                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)

                    yield dict(label=np.int32(label_id),wav=signal,)

                except Exception as err:
                    print(err, label_id, uid, fname)

        elif self.config.mode in ['test']:
            for path in self.test_data:
                signal = self._read_wav_and_pad(path)
                fname = os.path.basename(path)
                if self.config.mfcc:
                    signal = self._do_mfcc(signal)
                yield dict(label=np.string_(fname),wav=signal,)

        elif self.config.mode == 'background':
            data = []
            noise_fns = [self.config.dir_noise + fn for fn in os.listdir(self.config.dir_noise) if fn.endswith('.wav')]
            for fn in noise_fns:
                data.append((99, '', fn))
            np.random.shuffle(data)
            for (label_id, uid, fn) in data:
                try:
                    signal = self._read_wav_and_pad(fn)

                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)

                    yield dict(label=np.int32(label_id),wav=signal,)

                except Exception as err:
                    print(err, label_id, uid, fn)

        elif self.config.mode == 'unknown':
            data = self.train_data
            data = [d for d in data if d[0] == self.config.name2id['unknown']]
            print(len(data))
            np.random.shuffle(data)
            for (label_id, uid, fname) in data:
                try:
                    signal = self._read_wav_and_pad(fname)

                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)

                    yield dict(
                        label=np.int32(label_id),
                        wav=signal,
                    )

                except Exception as err:
                    print(err, label_id, uid, fname)
        elif self.config.mode == 'silence':
            data = []
            # append pure silence
            for _ in range(int(self.config.amount_silence * self.config.pure_silence_portion)):
                data.append((self.config.name2id['silence'], '', 'assets/data_augmentation/silence/pure_silence.wav'))

            noise_fns = os.listdir(self.config.dir_noise)
            noise_fns = [fn for fn in noise_fns if fn.endswith('.wav')]
            #append background noise as silence
            np.random.shuffle(noise_fns)
            end = int(self.config.amount_silence * (1-self.config.pure_silence_portion))
            for fn in noise_fns[:end]:
                data.append((self.config.name2id['silence'], '', self.config.dir_noise + fn))
            np.random.shuffle(data)
            # Feel free to add any augmentation
            for (label_id, uid, fname) in data:
                try:
                    signal = self._read_wav_and_pad(fname)
                    if self.config.noise2noise:
                        extra_noise = np.random.normal(1, 2, 16000)
                        volume_change = np.random.uniform(1,10)
                        signal = signal * extra_noise
                        signal = signal / volume_change
                    if self.config.mfcc:
                        signal = self._do_mfcc(signal)

                    yield dict(
                        label=np.int32(label_id),
                        wav=signal,
                    )

                except Exception as err:
                    print(err, label_id, uid, fname)

    def load_data(self):
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
                if label not in possible:

                    #ignore noise fns
                    if label == '_background_noise_':
                        continue

                    else:
                        label = 'unknown'


                label_id = self.config.name2id[label]
                sample = (label_id, uid, entry)
                if uid in valset:
                    val.append(sample)
                else:
                    train.append(sample)

        print('There are {} train and {} val samples'.format(len(train), len(val)))
        return train, val


    def build_train_and_val_corpus(self):
        self.config.mode = 'train'
        len_train = self.build_corpus()
        self.config.mode = 'valid'
        len_valid = self.build_corpus()
        return len_train, len_valid

    def build_corpus(self):
        corpus = []

        k = 0
        for date in self.data_gen():
            if k % 100 == 0:
                logging.debug('progress: ' + str(k) + '/' )
            corpus.append(date)
            k += 1
        save_name = self.config.save_dir
        save_name += self.config.mode + '.'
        save_name += ''.join(self.name_flags)
        save_name += '.soundcorpus.p'
        logging.info('saving under: ' + save_name)
        with open(save_name, 'wb') as f:
            pickler = pickle.Pickler(f)
            for e in corpus:
                pickler.dump(e)
        return len(corpus)

    ## UNUSED
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

        ## UNUSED
    @staticmethod
    def change_volume(wav,factor):
        wav = wav / factor
        return wav

if __name__ == '__main__':

    cfg_train = SC_Config(mode='train')
    train_corpus = SoundCorpusCreator(cfg_train)
    len_train = train_corpus.build_corpus()

    cfg_valid = SC_Config(mode='valid')
    cfg_valid.unknown_portion = 0.09
    valid_corpus = SoundCorpusCreator(cfg_valid)
    len_valid = valid_corpus.build_corpus()

    cfg_test = SC_Config(mode='test')
    test_corpus = SoundCorpusCreator(cfg_test)
    len_test = test_corpus.build_corpus()

    cfg_bg = SC_Config(mode='background')
    train_corpus = SoundCorpusCreator(cfg_bg)
    len_bg = train_corpus.build_corpus()

    cfg_unknown = SC_Config(mode='unknown')
    cfg_unknown.unknown_portion = 1
    unknown_corpus = SoundCorpusCreator(cfg_unknown)
    len_unknown = unknown_corpus.build_corpus()

    cfg_silence = SC_Config(mode='silence')
    silence_corpus = SoundCorpusCreator(cfg_silence)
    len_silence = silence_corpus.build_corpus()

    # should also save len of corpora
    info_dict = {'id2name': cfg_train.id2name,
                 'name2id': cfg_train.name2id,
                 'len_train': len_train,
                 'len_valid': len_valid,
                 'len_test': len_test,
                 'len_unknown':len_unknown,
                 'len_background':len_bg,
                 'len_silence':len_silence

                 }

    print(info_dict['id2name'])
    with open(cfg_train.save_dir + 'infos.p','wb') as f:
        pickle.dump(info_dict,f)