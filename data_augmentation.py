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
import matplotlib
matplotlib.use('TkAgg')
import os
import glob
import webrtcvad
import logging
import acoustics
from batch_gen import SoundCorpus

logging.basicConfig(level=logging.DEBUG)

bn_dir = 'assets/train/audio/_background_noise_/'
save_dir = 'assets/data_augmentation/silence/background/'
fns = [fn for fn in os.listdir(bn_dir) if not fn.startswith('.')]
fns = [fn for fn in fns if not fn.startswith('READ')]

seed = 24
np.random.seed(seed=seed)

def create_noise(fns, factor, lower_bound_silence= None, upper_bound_silence= None):
    for fn in fns:

        _, wav = wavfile.read(bn_dir + fn)
        #wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        L = 16000

        i = int(len(wav) / L) * factor
        for k in range(i):
            b = np.random.randint(1, len(wav) - L)
            wav_snip = wav[b:b + L]
            #wav_snip = wav_snip / np.random.uniform(lower_bound_silence,upper_bound_silence) #unneccessary
            wavfile.write(save_dir + fn[:-4] + str(b) + '.wav', L, wav_snip)


def get_noise_color(noise_color="white", is_float=False):
    if is_float:
        return np.array((acoustics.generator.noise(16000, color=noise_color))
                        / 3)
    else:
        return np.array((
            (acoustics.generator.noise(16000, color=noise_color)) / 3) *
                                    32767).astype(np.int16)


def read_wav(wav_name):
    fs, wav = wavfile.read(wav_name)
    #wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

def get_silence_audio(wav, sample_rate=16000, window_duration=0.03):
    vad_mode = 1
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    samples_per_window = int(window_duration * sample_rate + 0.5)
    n_segment = int(len(wav) / samples_per_window)
    new_wav = np.asarray([],dtype=np.int16)
    for i in range(n_segment-1):
        #logging.log(logging.DEBUG,"segment:"+str(i)+"/"+str(n_segment))
        start = i * samples_per_window
        stop = (i+1) * samples_per_window
        wav_bytes = wav[start:stop].tobytes()
        is_speech = vad.is_speech(wav_bytes, sample_rate=sample_rate)
        if not is_speech:
            segmented_wav = wav[start:stop]
            new_wav = np.concatenate((new_wav, segmented_wav), axis=0)
    return new_wav


def add_noise(wav, noise_color='white', noise_ratio=0.5):
    noise = get_noise_color(noise_color, is_float= True)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    wav = (((1-noise_ratio) + noise_ratio*noise[:len(wav)]) * np.iinfo(
        np.int16).max).astype(np.int16)
    return wav

def create_silence():
    train_dir = 'assets/train/audio/'
    save_dir2 = 'assets/data_augmentation/silence/artificial_silence/'
    all_training_files = glob.glob(os.path.join(train_dir, '*', '*.wav'))
    speech_training_files = [x for x in all_training_files if not
    os.path.dirname(x) + "/" == bn_dir]
    L = 16000
    n = len(speech_training_files)
    noise_color_list = ["white", "pink", "blue", "brown", "violet"]
    noise_ratio = 0.5
    is_add_noise = True
    for i,wav_files in enumerate(speech_training_files):
        logging.log(logging.DEBUG,"wav:"+str(i)+"/"+str(n))
        wav_name = os.path.basename(wav_files)
        dir_name = os.path.basename(os.path.dirname(wav_files))
        wav = read_wav(wav_files)
        new_wav = get_silence_audio(wav)
        len_new_wav = len(new_wav)
        if len_new_wav < L:
            new_wav = np.concatenate((new_wav, np.full((L-len_new_wav), 0,dtype=float)), axis=0)
        elif len_new_wav > L:
            new_wav = new_wav[:L]
        if is_add_noise:
            noise_color = np.random.choice(noise_color_list, 1)[0]
            new_wav = add_noise(new_wav,noise_color, noise_ratio)
            print(wav_name, len(new_wav), noise_color)
        else:
            print(wav_name, len(new_wav))
        new_wav_name = dir_name + "_" + wav_name
        wavfile.write(os.path.join(save_dir2, new_wav_name), L,
                      new_wav.astype(np.float32))

def create_silence2():
    train_corpus = SoundCorpus('assets/corpora/corpus14/', mode='train')
    save_dir = 'assets/data_augmentation/silence/artificial_silence/'
    data = [d for d in train_corpus]
    new_silence = np.asarray([],dtype=np.int16)
    for k,d in enumerate(data):
        print(k)
        wav = np.int16(d['wav'] * 2**15)
        try:
            new_wav = get_silence_audio(wav)
            new_silence = np.concatenate((new_silence, new_wav), axis=0)
        except:
            continue
    for k in range(int(len(new_silence)/16000)):
        new_wav = new_silence[k*16000:(k+1)*16000]
        new_wav_name = 'art_silence' + str(k) + '.wav'
        wavfile.write(os.path.join(save_dir, new_wav_name), 16000,new_wav)

    return new_silence

def create_silence3():
    train_dir = 'assets/train/audio/'
    save_dir2 = 'assets/data_augmentation/silence/artificial_silence4/'
    all_training_files = glob.glob(os.path.join(train_dir, '*', '*.wav'))
    speech_training_files = [x for x in all_training_files if not
    os.path.dirname(x) + "/" == bn_dir]
    np.random.seed(1)
    L = 16000
    n = len(speech_training_files)
    noise_color_list = ["white", "pink", "blue", "brown", "violet"]
    silence_part_port = 0.05
    new_wav = np.asarray([], np.int16)
    for i,wav_files in enumerate(speech_training_files):

        logging.log(logging.DEBUG,"wav:"+str(i)+"/"+str(n))
        wav_name = os.path.basename(wav_files)
        dir_name = os.path.basename(os.path.dirname(wav_files))
        fs, wav = wavfile.read(wav_files)
        silence_part = wav[:int(L*silence_part_port)]
        if len(new_wav) > L:
            new_wav = new_wav[:L]
            noise_color = np.random.choice(noise_color_list, 1)[0]
            factor_mix = np.random.uniform(0.3, 0.8)
            new_wav = add_noise(new_wav, noise_color, noise_ratio=factor_mix)
            new_wav_name = dir_name + "_" + wav_name
            wavfile.write(os.path.join(save_dir2, new_wav_name), L,
                          new_wav)
            new_wav = np.asarray([], np.int16)
        else:
            new_wav = np.concatenate((new_wav,silence_part),axis=0)


def create_unknown():
    train_corpus = SoundCorpus('assets/corpora/corpus2/', mode='unknown')
    save_dir = 'assets/data_augmentation/unknown/artificial_unknown/'
    data = [np.int16(d['wav'] * 2**15) for d in train_corpus]
    parts = 10
    ids = [id for id in range(parts)]
    n = 0
    for k in range(int(len(data)/parts)):
        print(k)
        data0 = data[parts*k:parts*(k+1)]


        len_part = int(16000/parts)

        for w in range(parts):
            np.random.shuffle(ids)
            new_wav = np.asarray([],dtype=np.int16)
            for p in range(parts):
                new_part = data0[ids[p]][len_part*p:len_part*(p+1)]
                new_wav = np.concatenate((new_wav, new_part), axis=0)
            new_wav_name = 'art_unknown' + str(n) + '.wav'
            wavfile.write(save_dir + new_wav_name, 16000,new_wav)
            n+=1

if __name__ == '__main__':
    #create_unknown()
    #create_noise(fns,10)
    #create_silence2()
    create_silence3()
