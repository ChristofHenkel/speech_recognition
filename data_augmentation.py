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
import glob
import webrtcvad
import struct
from silence_detection import *

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


 #
 # def is_silence(wav):
 #
 #
 #        samples_per_window = int(window_duration * 16000 + 0.5)
 #        bytes_per_sample = 2
 #        speech_analysis = []
 #        for start in np.arange(0, len(wav), samples_per_window):
 #            stop = min(start + samples_per_window, len(wav))
 #            is_speech = self.vad.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample],
 #                                      sample_rate=16000)
 #            speech_analysis.append(is_speech)
 #
 #        speech_port = speech_analysis.count(True) / len(speech_analysis)
 #        return speech_port < self.speech_portion_threshold

def process_speech2silence(wav, sample_rate=16000, window_duration=0.3):
    vad_mode = 1
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    samples_per_window = int(window_duration * sample_rate + 0.5)
    n_segment = int(window_duration / samples_per_window)
    segmented_wav = []
    new_wav=[]
    for i in range(n_segment):
        segmented_wav[i] = wav[i*samples_per_window:(i+1)*samples_per_window]
    is_speech = [vad.is_speech(vad.is_speech(x, sample_rate=sample_rate) for x in
                               segmented_wav)]
    for i,speech in enumerate(is_speech):
        if not speech:
            new_wav= np.concatenate(new_wav,segmented_wav[i])
    return new_wav

def create_silence():
    #bn_dir = 'assets/train/audio/_background_noise_/'
    sd = SilenceDetector()
    train_dir = 'assets/train/audio/'
    save_dir2 = 'assets/data_augmentation/silence/concatenate/'
    all_training_files = glob.glob(os.path.join(train_dir,'*','*.wav'))
    speech_training_files = [x for x in all_training_files if not
    os.path.dirname(x) + "/" == bn_dir]
    L = 16000
    for wav_files in speech_training_files:
        wav_name = os.path.basename(wav_files)
        dir_name = os.path.basename(os.path.dirname(wav_files))
        wav = sd._read_wav_and_pad(wav_files)
        new_wav = process_speech2silence(wav)
        new_wav_name = dir_name + "_" + wav_name
        wavfile.write(os.path.join(save_dir2, new_wav_name, L, new_wav))


if __name__ == '__main__':

    # create_noise(fns,10,1,2)
    create_silence()