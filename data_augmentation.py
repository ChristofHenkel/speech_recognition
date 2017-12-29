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
import logging
import acoustics

logging.basicConfig(level=logging.DEBUG)
train_dir = 'assets/train/audio/'
test_dir = 'assets/test/audio/'
bn_dir = 'assets/train/audio/_background_noise_/'
save_dir_background = 'assets/data_augmentation/silence/background/'
save_dir_silence = 'assets/data_augmentation/silence/artificial_silence/'
save_dir_unknown = 'assets/data_augmentation/unknown/artificial_unknown/'
possible_labels = 'yes no up down left right on off stop go unknown silence'.split()


background_fns = [fn for fn in os.listdir(bn_dir) if not fn.startswith('.')]
background_fns = [fn for fn in background_fns if not fn.startswith('READ')]

seed = 1
np.random.seed(seed=seed)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def filter_unknown(speech_files):
    fns = [fn for fn in speech_files if fn.split('/')[-2] not in possible_labels]
    return fns


def load_speech_files(include_test_files = True):
    all_training_files = glob.glob(os.path.join(train_dir, '*', '*.wav'))

    speech_files = [x for x in all_training_files if not os.path.dirname(x) + "/" == bn_dir]
    if include_test_files:
        all_test_files = glob.glob(os.path.join(test_dir, '*.wav'))
        speech_files.extend(all_test_files)
    return speech_files

def split_background_files(fns, factor):
    for fn in fns:

        _, wav = wavfile.read(bn_dir + fn)
        L = 16000

        i = int(len(wav) / L) * factor
        for k in range(i):
            b = np.random.randint(1, len(wav) - L)
            wav_snip = wav[b:b + L]
            wavfile.write(save_dir_background + fn[:-4] + str(b) + '.wav', L, wav_snip)


def get_noise_color(noise_color="white", is_float=False, bn_fn = None):
    if is_float:
        if noise_color != 'background':
            return np.array((acoustics.generator.noise(16000, color=noise_color))
                        / 3)
        else:
            fn = np.random.choice(bn_fn)
            wav = read_wav(save_dir_background + fn)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            return wav


    else:
        return np.array((
            (acoustics.generator.noise(16000, color=noise_color)) / 3) *
                                    32767).astype(np.int16)


def read_wav(wav_name):
    fs, wav = wavfile.read(wav_name)
    len_wav = len(wav)
    if len_wav < 16000:  # be aware, some files are shorter than 1 sec!
        padded = np.zeros([16000], dtype=np.int16)
        start = np.random.randint(0, 16000 - len_wav)
        end = start + len_wav
        padded[start:end] = wav
        wav = padded
    if len_wav > 16000:
        print(len_wav)
        beg = np.random.randint(0, len_wav - 16000)
    else:
        beg = 0

    signal = wav[beg: beg + 16000]
    return signal


def add_noise(wav, noise_color, noise_ratio):
    noise = get_noise_color(noise_color, is_float= True, bn_fn = os.listdir(save_dir_background))

    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    wav = (((1-noise_ratio)*wav + noise_ratio*noise[:len(wav)]) * np.iinfo(np.int16).max).astype(np.int16)
    return wav


def create_silence(speech_files):


    #np.random.seed(1)
    L = 16000
    n = len(speech_files)
    noise_color_list = ["blue", "brown", "violet", "background"]
    silence_part_port = 0.02
    new_wav = np.asarray([], np.int16)
    for i,wav_files in enumerate(speech_files):

        logging.log(logging.DEBUG,"wav:"+str(i)+"/"+str(n))
        wav_name = os.path.basename(wav_files)
        dir_name = os.path.basename(os.path.dirname(wav_files))
        wav = read_wav(wav_files)
        silence_part = np.concatenate((wav[:int(L*silence_part_port)],wav[int(L*(1-silence_part_port)):]), axis=0)
        if len(new_wav) > L:
            new_wav = new_wav[:L]
            noise_color = np.random.choice(noise_color_list, 1, p=[0.1,0.1,0.1,0.7])[0]
            factor_mix = np.random.uniform()
            #factor_mix = np.exp(-x)
            new_wav = add_noise(new_wav, noise_color, noise_ratio=factor_mix)
            new_wav_name = dir_name + "_" + wav_name
            wavfile.write(os.path.join(save_dir_silence, new_wav_name), L,
                          new_wav)
            new_wav = np.asarray([], np.int16)
        else:
            new_wav = np.concatenate((new_wav,silence_part),axis=0)


def create_unknown(speech_files):
    fns = filter_unknown(speech_files)


    data = [read_wav(fn) for fn in fns]
    np.random.shuffle(data)
    parts = 3
    ids = [id for id in range(parts)]
    n = 0
    iters = int(len(data)/parts)
    for k in range(iters):
        print('%s / %s' %(k,iters))
        data0 = data[parts*k:parts*(k+1)]


        len_part = int(16000/parts)

        #for w in range(parts): #could make 3 files of 3 files but we`ll use anyway max 30% augmented
        np.random.shuffle(ids)
        new_wav = np.asarray([],dtype=np.int16)
        for p in range(parts):
            new_part = data0[ids[p]][len_part*p:len_part*(p+1)]
            new_wav = np.concatenate((new_wav, new_part), axis=0)
        padded_new_wav = np.zeros(16000,dtype=np.int16)
        padded_new_wav[:new_wav.shape[0]] = new_wav
        new_wav_name = 'art_unknown' + str(n) + '.wav'
        wavfile.write(save_dir_unknown + new_wav_name, 16000,padded_new_wav)
        n+=1

if __name__ == '__main__':
    ensure_dir(save_dir_background)
    ensure_dir(save_dir_silence)
    ensure_dir(save_dir_unknown)

    split_background_files(background_fns,2)
    speech_files = load_speech_files()
    create_silence(speech_files)
    train_files = load_speech_files(include_test_files=False)
    create_unknown(train_files)
    seed_ckpt = np.random.randint(1,999)
    if seed_ckpt == 979:
        print('seed check ok')
    else:
        print('seed not consistent')
