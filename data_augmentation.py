"""
TODO
# Shifting the sound
data_roll = np.roll(data, 1600)

# stretching the sound
def stretch(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

"""

from scipy.io import wavfile
import numpy as np
import os

bn_dir = 'assets/train/audio/_background_noise_/'
save_dir = 'assets/data_augmentation/silence/background/'
fns = [fn for fn in os.listdir(bn_dir) if not fn.startswith('.')]
fns = [fn for fn in fns if not fn.startswith('READ')]

def create_noise(fns,factor,lower_bound_silence,upper_bound_silence):

    for fn in fns:

        _, wav = wavfile.read(bn_dir + fn)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        L = 16000

        i = int(len(wav)/L)*factor
        for k in range(i):
            b = np.random.randint(1,len(wav)-L)
            wav_snip = wav[b:b+L]
            wav_snip = wav_snip / np.random.uniform(lower_bound_silence,upper_bound_silence)
            wavfile.write(save_dir + fn[:-4] + str(b) + '.wav',L,wav_snip)


if __name__ == '__main__':

    create_noise(fns,10,1,2)