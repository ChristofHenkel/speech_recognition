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
import logging
import acoustics

logging.basicConfig(level=logging.DEBUG)

bn_dir = 'assets/train/audio/_background_noise_/'
save_dir = 'assets/data_augmentation/silence/background/'
fns = [fn for fn in os.listdir(bn_dir) if not fn.startswith('.')]
fns = [fn for fn in fns if not fn.startswith('READ')]


def create_noise(fns, factor, lower_bound_silence, upper_bound_silence):
    for fn in fns:

        _, wav = wavfile.read(bn_dir + fn)
        wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        L = 16000

        i = int(len(wav) / L) * factor
        for k in range(i):
            b = np.random.randint(1, len(wav) - L)
            wav_snip = wav[b:b + L]
            wav_snip = wav_snip / np.random.uniform(lower_bound_silence,
                                                    upper_bound_silence)
            wavfile.write(save_dir + fn[:-4] + str(b) + '.wav', L, wav_snip)

def get_noise_color(noise_color="white"):
    return np.array((
        (acoustics.generator.noise(16000 * 60, color=noise_color)) / 3) *
                                32767).astype(np.int16)

def get_silence_audio(wav_name, sample_rate=16000, window_duration=0.01):
    fs, wav = wavfile.read(wav_name)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    vad_mode = 1
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    samples_per_window = int(window_duration * sample_rate + 0.5)
    n_segment = int(len(wav) / samples_per_window)
    new_wav = np.asarray([])
    for i in range(n_segment-1):
        #logging.log(logging.DEBUG,"segment:"+str(i)+"/"+str(n_segment))
        start = i * samples_per_window
        stop = (i+1) * samples_per_window
        wav_bytes = wav[start:stop].tobytes()
        is_speech = vad.is_speech(wav_bytes, sample_rate=sample_rate)
        if not is_speech:
            segmented_wav = np.asarray(wav[start:stop])
            new_wav = np.concatenate((new_wav, segmented_wav), axis=0)
    return new_wav


def add_noise(wav, noise_color='white', ratio=0.5):
    noise = get_noise_color(noise_color)
    noise = noise.astype(np.float32) / np.iinfo(np.int16).max
    wav = wav + (ratio*noise[:len(wav)])
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
        new_wav = get_silence_audio(wav_files)
        len_new_wav = len(new_wav)
        if len_new_wav < L:
            new_wav = np.concatenate((new_wav, np.full((L-len_new_wav), 0,
                                                       dtype=float)), axis=0)
        elif len_new_wav > L:
            new_wav = new_wav[:L]
        if is_add_noise:
            noise_color = np.random.choice(noise_color_list, 1)[0]
            new_wav = add_noise(new_wav,noise_color, noise_ratio)
        new_wav_name = dir_name + "_" + wav_name
        if is_add_noise:
            print(wav_name, len(new_wav), noise_color)
        else:
            print(wav_name, len(new_wav))
        wavfile.write(os.path.join(save_dir2, new_wav_name), L,
                      new_wav.astype(np.float32))


if __name__ == '__main__':
    create_noise(fns,10,1,2)
    create_silence()
