import os
import pickle
import numpy as np
from python_speech_features import mfcc, delta
import sounddevice as sd
from input_features import stacked_mfcc

class SoundCorpus:

    def __init__(self, soundcorpus_dir, mode, fn = None):
        self.mode = mode
        assert self.mode in ['train','own_test','test','unknown','background','silence','valid']
        if fn is None:
            self.fp = soundcorpus_dir + [fn for fn in os.listdir(soundcorpus_dir) if fn.startswith(self.mode)][0]
        else:
            self.fp = soundcorpus_dir + fn
        self.info_dict_fp = soundcorpus_dir + 'infos.p'
        self.decoder = None
        self.encoder = None
        self.len = None
        if not mode == 'own_test':
            self._load_info_dict()
        self.gen = self.__iter__()

    def __iter__(self):
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                try:
                    item = unpickler.load()
                    yield item
                except EOFError:
                    break

    def _load_info_dict(self):
        with open(self.info_dict_fp,'rb') as f:
            content = pickle.load(f)
            self.decoder = content['id2name']
            self.encoder = content['name2id']
            self.len = content['len_' + self.mode]

    def _get_len(self):
        size = 0
        for _ in self.gen:
            size +=1
        return size

    def _get_last_n(self,length, n):
        pos = 0
        items = []
        for item in self.gen:
            pos +=1
            if pos > (length-n):
                items.append(item)
        return items

    def play_next(self):
        data = next(self.gen)
        wav = data['wav']
        print('label: ',str(data['label']))
        sd.play(wav,16000,blocking = True)

    def batch_gen(self,batch_size, do_mfcc = False, mfcc_dims = None):
        x = []
        y = []
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                #try:
                item = unpickler.load()
                wav = item['wav']
                label = item['label']
                if do_mfcc:
                    wav = stacked_mfcc(wav,numcep=mfcc_dims[1], num_layers=mfcc_dims[2])

                x.append(wav)
                y.append(label)

                if len(x) == batch_size:
                    # reshape to np arrays
                    x = np.asarray(x)
                    if not self.mode in ['test']:
                        y = np.asarray(y)
                    yield x, y
                    x = []
                    y = []


class BatchGenerator:

    def __init__(self, BatchParams, TrainCorpus=None, BackgroundCorpus=None, UnknownCorpus=None, SilenceCorpus=None):
        self.batch_size = BatchParams.batch_size
        self.train_corpus = TrainCorpus
        self.background_corpus = BackgroundCorpus
        self.unknown_corpus = UnknownCorpus
        self.silence_corpus = SilenceCorpus
        self.portion_unknown = BatchParams.portion_unknown
        self.portion_silence = BatchParams.portion_silence
        self.portion_noised = BatchParams.portion_noised
        self.lower_bound_noise_mix = BatchParams.lower_bound_noise_mix
        self.upper_bound_noise_mix = BatchParams.upper_bound_noise_mix
        self.noise_unknown = BatchParams.noise_unknown
        self.noise_silence = BatchParams.noise_silence
        self.do_mfcc = BatchParams.do_mfcc
        self.all_gen = self.batch_gen()
        self.dims_mfcc = BatchParams.dims_mfcc

        self.batches_counter = 0


    @staticmethod
    def _combine_wav(wav1,wav2,factor_wav1, wav1_is_silence = False):
        if wav1_is_silence:
            combined_wav = wav1 + factor_wav1 * wav2
        else:
            try:
                combined_wav = factor_wav1 * wav1 + (1-factor_wav1) * wav2
            except:
                print(factor_wav1)
                print(wav1.shape)
                print(wav2.shape)
                combined_wav = wav1
        return combined_wav

    def gen_corpus(self,fp):
        with open(fp, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while True:
                data = unpickler.load()
                yield data


    def batch_gen(self):
        x = []
        y = []
        gen_train = self.gen_corpus(self.train_corpus.fp)
        gen_noise = self.gen_corpus(self.background_corpus.fp)
        gen_unknown = self.gen_corpus(self.unknown_corpus.fp)
        # gen_silence = self.gen_corpus(self.silence_corpus.fp)
        gen_silence = None

        while True:
            type = np.random.choice(['known', 'unknown', 'silence'],
                                    p=[1 - self.portion_unknown - self.portion_silence,
                                       self.portion_unknown,
                                       self.portion_silence])

            if type == 'known':
                try:
                    train_data = next(gen_train)
                except EOFError:
                    print('restarting gen_train')
                    gen_train = self.gen_corpus(self.train_corpus.fp)
                    train_data = next(gen_train)
            elif type == 'unknown':
                try:
                    train_data = next(gen_unknown)

                except EOFError:
                    print('restarting gen_unknown')
                    gen_unknown = self.gen_corpus(self.unknown_corpus.fp)
                    train_data = next(gen_unknown)
            else:
                try:
                    train_data = next(gen_silence)

                except EOFError:
                    print('restarting gen_silence')
                    gen_silence = self.gen_corpus(self.silence_corpus.fp)
                    train_data = next(gen_silence)
            try:
                noise = next(gen_noise)
            except EOFError:
                print('restarting gen_bg')
                gen_noise = self.gen_corpus(self.background_corpus.fp)
                noise = next(gen_noise)

            raw_wav = train_data['wav']
            noise_wav = noise['wav']
            label = train_data['label']
            factor_mix = 1- np.random.uniform(self.lower_bound_noise_mix,self.upper_bound_noise_mix)
            if np.random.rand() < self.portion_noised:
                if type is 'silence':
                    if self.noise_silence:
                        wav = self._combine_wav(raw_wav, noise_wav, factor_mix)
                    else:
                        wav = raw_wav
                elif type is 'known':
                    wav = self._combine_wav(raw_wav, noise_wav, factor_mix)
                else:
                    if self.noise_unknown:
                        wav = self._combine_wav(raw_wav, noise_wav, factor_mix)
                    else:
                        wav = raw_wav
            else:
                wav = raw_wav
            if self.do_mfcc:
                signal = stacked_mfcc(wav,num_layers=self.dims_mfcc[2],numcep=self.dims_mfcc[1])
            else:
                signal = wav
            x.append(signal)
            y.append(label)
            if len(x) == self.batch_size:
                x = np.asarray(x)
                yield x, y
                self.batches_counter += 1
                x = []
                y = []

    # not tested yet
    def _play_next(self,index):
        x,y = next(self.all_gen)
        wav = x[index]
        label = y[index]
        print(label)
        sd.play(wav, 16000, blocking=True)