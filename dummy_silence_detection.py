import glob
import os
import numpy as np
from scipy.io import wavfile
import logging
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from python_speech_features import mfcc, logfbank, delta

logging.basicConfig(level=logging.DEBUG)

class SilenceDetector:
    def __init__(self):
        self.data_root = 'assets/'
        self.dir_files = 'train/audio/*/*wav'
        self.L = 16000  # length of files
        self.config_padding = True

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
        len_wav = wav.shape[0]
        if len_wav < self.L:    # be aware, some files are shorter than 1 sec!
            if self.config_padding:
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

    def process(self):
        #for fname in glob.glob(os.path.join(self.data_root, self.dir_files)):
            #print(fname)
        sil_fname = "/Users/maureen/Documents/Work/kaggle/assets/data_augmentation/silence/background/doing_the_dishes1522.wav"
        dog_fname = "/Users/maureen/Documents/Work/kaggle/assets/train/audio/dog/0a7c2a8d_nohash_0.wav"
        sil_signal = self._read_wav_and_pad(sil_fname)
        dog_signal = self._read_wav_and_pad(dog_fname)
        # sil_signal = self._do_mfcc(sil_signal)
        # dog_signal = self._do_mfcc(dog_signal)
        sil_signal = np.square(sil_signal)
        dog_signal = np.square(dog_signal)
        fs = 16000
        analytic_signal = hilbert(sil_signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
        fig = plt.figure()
        ax0 = fig.add_subplot(211)
        ax0.plot(t, signal, label='signal')
        ax0.plot(t, amplitude_envelope, label='envelope')
        ax0.set_xlabel("time in seconds")
        ax0.legend()
        ax1 = fig.add_subplot(212)
        ax1.plot(t[1:], instantaneous_frequency)
        ax1.set_xlabel("time in seconds")
        ax1.set_ylim(0.0, 120.0)

if __name__ == "__main__":
    SC = SilenceDetector()
    SC.process()
