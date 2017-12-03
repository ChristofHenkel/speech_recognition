from python_speech_features import mfcc, delta
import numpy as np

# signal is from tensorflow

# 1)
# raw audio


# 2)

#specgram = signal.stft(
#    wav,
#    400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
#    160,  # 16000 * 0.010 -- default stride
#)
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

def stacked_mfcc(signal):
    signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                  nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                  ceplifter=22, appendEnergy=True)
    dsignal = delta(signal, N=1)
    ddsignal = delta(dsignal, N=1)
    signal = np.stack([signal, dsignal, ddsignal], axis=2)
    return signal

#stacked mfcc

# mfcc
# delta
# deltadelta