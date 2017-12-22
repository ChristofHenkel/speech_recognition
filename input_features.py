from python_speech_features import mfcc, delta, logfbank
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

def stacked_mfcc(signal, num_layers = 3, numcep = 13):
    signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=numcep,
                  nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                  ceplifter=22, appendEnergy=True, winfunc=np.hamming)
    signal -= (np.mean(signal, axis=0) + 1e-8)
    signal_stack = np.expand_dims(signal, axis=2)

    for k in range(num_layers-1):
        delta_ = delta(signal_stack[:,:,k],N=1)
        delta_ = np.expand_dims(delta_, axis=2)
        signal_stack = np.concatenate((signal_stack, delta_), axis=2)


    return signal_stack

# 5)

def stacked_filterbank(signal, num_layers = 1, fs=16000, winlen=0.025, winstep=0.01, nfilt=26,
                    nfft=512, preemph=0.97):
    signal = logfbank(signal, samplerate=fs, winlen=winlen, winstep=winstep,
                      nfilt=nfilt, nfft=nfft, lowfreq=0, highfreq=None,
                      preemph=preemph)
    signal -= (np.mean(signal, axis=0) + 1e-8)
    signal_stack = np.expand_dims(signal, axis=2)

    for k in range(num_layers-1):
        delta_ = delta(signal_stack[:,:,k],N=1)
        delta_ = np.expand_dims(delta_, axis=2)
        signal_stack = np.concatenate((signal_stack, delta_), axis=2)

    return signal_stack


# 6)
# Maureen: GMM

#if __name__ == "__main__":
