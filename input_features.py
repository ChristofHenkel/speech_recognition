from python_speech_features import mfcc, delta, logfbank
import tensorflow as tf
from scipy import signal
import numpy as np

# signal is from tensorflow

# 1)
# raw audio


# 2)

def get_spectrogram(signals, fs=16000, winlen=0.025, winstep=0.1, nfft=512):
    nooverlap = fs * winstep
    window_func = ('tukey', winlen)
    f, t, Sxx = signal.spectrogram(signals, fs=fs, window=window_func,
        noverlap=nooverlap, nfft=nfft)
    return f, t, Sxx


def get_amplitude_phase(specgram):
    phase = np.angle(specgram) / np.pi
    amp = np.log1p(np.abs(specgram))
    return phase, amp


## specgram is a complex tensor, so split it into abs and phase parts:
#phase = tf.angle(specgram) / np.pi
## log(1 + abs) is a default transformation for energy units
#amp = tf.log1p(tf.abs(specgram))
#
#x = tf.stack([amp, phase], axis=3)

# 3)
# Christof also from audio sentiment analysis I have done previously, I have a collection of input features.
# need to look it up

# 4)

def stacked_mfcc(signals, fs=16000, winlen=0.25, winstep=0.1,
                    nfft=512, preemph=0.97):
    signals = mfcc(signals, samplerate=fs, winlen=winlen, winstep=winstep,
                  numcep=13, nfilt=26, nfft=nfft, lowfreq=0, highfreq=None,
                  preemph=preemph, ceplifter=22, appendEnergy=True)
    dsignals = delta(signals, N=1)
    ddsignals = delta(dsignals, N=1)
    signal = np.stack([signals, dsignals, ddsignals], axis=2)
    return signal


# 5)

def log_filter_bank(signals, fs=16000, winlen=0.25, winstep=0.1, nfilt=26,
                    nfft=512, preemph=0.97):
    log_fb = logfbank(signals, samplerate=fs, winlen=winlen, winstep=winstep,
                      nfilt=nfilt, nfft=nfft, lowfreq=0, highfreq=None,
                      preemph=preemph)
    return log_fb


# 6)
# Maureen: GMM

#if __name__ == "__main__":
