import os
import pickle
import numpy as np
from python_speech_features import mfcc, delta

class SoundCorpus:

    def __init__(self, soundcorpus_dir, mode = 'train', fn = None):
        self.mode = mode
        if fn is None:
            self.fp = soundcorpus_dir + [fn for fn in os.listdir(soundcorpus_dir) if fn.startswith(self.mode)][0]
        else:
            self.fp = soundcorpus_dir + fn
        self.info_dict_fp = soundcorpus_dir + 'infos.p'
        self.decoder = None
        self.encoder = None
        self.len = None
        self._load_info_dict()


    def _load_info_dict(self):
        with open(self.info_dict_fp,'rb') as f:
            content = pickle.load(f)
            self.decoder = content['id2name']
            self.encoder = content['name2id']
            self.len = content['len_' + self.mode]

    def batch_gen(self,batch_size):
        x = []
        y = []
        with open(self.fp, 'rb') as file:
            unpickler = pickle.Unpickler(file)
            while True:
                #try:
                item = unpickler.load()

                x.append(item['wav'])
                y.append(item['label'])

                if len(x) == batch_size:
                    # reshape to np arrays
                    x = np.asarray(x)
                    if not self.mode in ['test']:
                        y = np.asarray(y)
                    yield x, y
                    x = []
                    y = []


class BatchGenerator:

    def __init__(self, TrainCorpus, BackgroundCorpus, UnknownCorpus,SilenceCorpus, batch_size):
        self.batch_size = batch_size
        self.train_corpus = TrainCorpus
        self.background_corpus = BackgroundCorpus
        self.unknown_corpus = UnknownCorpus
        self.silence_corpus = SilenceCorpus
        self.lower_bound_noise_mix = 0.6
        self.upper_bound_noise_mix = 1
        self.portion_unknown = 0.1
        self.portion_silence = 0.1
        self.portion_noised = 0.5
        self.noise_unknown = False
        self.noise_silence = True

    @staticmethod
    def _combine_wav(wav1,wav2,factor_wav1, wav1_is_silence = False):
        if  wav1_is_silence:
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

    @staticmethod
    def _do_mfcc(signal):
        signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                      nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                      ceplifter=22, appendEnergy=True)
        dsignal = delta(signal, N=1)
        ddsignal = delta(dsignal, N=1)
        signal = np.stack([signal, dsignal, ddsignal], axis=2)
        return signal

    def gen_train(self):
        with open(self.train_corpus.fp, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while True:
                data = unpickler.load()
                yield data

    def gen_bg(self):
        with open(self.background_corpus.fp, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while True:
                data = unpickler.load()
                yield data

    def gen_unknown(self):
        with open(self.unknown_corpus.fp, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while True:
                data = unpickler.load()
                yield data

    def gen_silence(self):
        with open(self.silence_corpus.fp, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while True:
                data = unpickler.load()
                yield data

    def batch_gen(self):
        x = []
        y = []
        gen_train = self.gen_train()
        gen_noise = self.gen_bg()
        gen_unknown = self.gen_unknown()
        gen_silence = self.gen_silence()
        while True:
            type = np.random.choice(['known', 'unknown', 'silence'],
                                    p=[1 - self.portion_unknown + self.portion_silence,
                                       self.portion_unknown,
                                       self.portion_silence])

            if type == 'known':
                try:
                    train_data = next(gen_train)
                except EOFError:
                    print('restarting gen_train')
                    gen_train = self.gen_train()
                    train_data = next(gen_train)
            elif type == 'unknown':
                try:
                    train_data = next(gen_unknown)

                except EOFError:
                    print('restarting gen_unknown')
                    gen_unknown = self.gen_unknown()
                    train_data = next(gen_unknown)
            else:
                try:
                    train_data = next(gen_silence)

                except EOFError:
                    print('restarting gen_silence')
                    gen_unknown = self.gen_silence()
                    train_data = next(gen_silence)
            try:
                noise = next(gen_noise)
            except EOFError:
                print('restarting gen_bg')
                gen_noise = self.gen_bg()
                noise = next(gen_noise)

            raw_wav = train_data['wav']
            noise_wav = noise['wav']
            label = train_data['label']
            factor_mix = np.random.uniform(self.lower_bound_noise_mix,self.upper_bound_noise_mix)
            if np.random.rand() > self.portion_noised:
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
            signal = self._do_mfcc(wav)
            x.append(signal)
            y.append(label)
            if len(x) == self.batch_size:
                x = np.asarray(x)
                yield x, y
                x = []
                y = []

