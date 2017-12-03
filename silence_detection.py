import glob
import os
import numpy as np
from scipy.io import wavfile
import logging
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
from python_speech_features import mfcc, logfbank, delta
import acoustics
from scipy.signal import butter, lfilter, freqz, fftconvolve, welch
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)


class SilenceDetector:
    def __init__(self):
        self.data_root = 'assets/'
        self.dir_files = 'train/audio/*/*wav'
        self.L = 16000  # length of files
        self.config_padding = True
        self.white_noise = None
        self.pink_noise = None

    def config(self):
        self.white_noise = np.array((
            (acoustics.generator.noise(16000 * 60, color='white')) / 3) *
                                    32767).astype(np.int16)
        self.pink_noise = np.array((
           (acoustics.generator.noise(16000 * 60, color='pink')) / 3) *
                                   32767).astype(np.int16)

    @staticmethod
    def _do_mfcc(signal):
        signal = mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                      numcep=13,
                      nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                      preemph=0.97,
                      ceplifter=22, appendEnergy=True)
        dsignal = delta(signal, N=1)
        ddsignal = delta(dsignal, N=1)
        signal = np.stack([signal, dsignal, ddsignal], axis=2)
        return signal

    def _read_wav_and_pad(self, fname):
        _, wav = wavfile.read(fname)
        len_wav = wav.shape[0]
        if len_wav < self.L:  # be aware, some files are shorter than 1 sec!
            if self.config_padding:
                # randomly insert wav into a 16k zero pad
                padded = np.zeros([self.L])
                start = np.random.randint(0, self.L - len_wav)
                end = start + len_wav
                padded[start:end] = wav
                wav = padded
        if len_wav > self.L:
            beg = np.random.randint(0, len_wav - self.L)
        else:
            beg = 0

        signal = wav[beg: beg + self.L]
        return signal

    @staticmethod
    def get_amplitude_envelop(signal, fs=16000, duration=1.0, is_plot=False):
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (
                np.diff(instantaneous_phase) / (2.0 * np.pi) * fs)
        if is_plot:
            samples = int(fs * duration)
            t = np.arange(samples) / fs
            fig = plt.figure()
            ax0 = fig.add_subplot(211)
            ax0.plot(t, signal, label='signal')
            ax0.plot(t, amplitude_envelope, label='envelope')
            ax0.set_xlabel("time in seconds")
            ax0.legend()
            ax1 = fig.add_subplot(212)
            ax1.plot(instantaneous_frequency)
            ax1.set_xlabel("time in seconds")
            ax1.set_ylim(0.0, 120.0)
        return amplitude_envelope

    @staticmethod
    def butter_bandpass(low, high, fs, order=5):
        nyq = 0.5 * fs
        low_cutoff = low / nyq
        high_cutoff = high / nyq
        b, a = butter(order, [low_cutoff, high_cutoff], btype='band',
                      analog=False)
        return b, a

    def butter_bandpass_filter(self, data, low, high, fs, order=5):
        b, a = self.butter_bandpass(low, high, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def demo_lf(self):
        # Filter requirements.
        order = 6
        fs = 30.0  # sample rate, Hz
        cutoff = 3.667  # desired cutoff frequency of the filter, Hz

        # Get the filter coefficients so we can check its frequency response.
        b, a = self.butter_lowpass(cutoff, fs, order)

        # Plot the frequency response.
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5 * fs)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        # Demonstrate the use of the filter.
        # First make some data to be filtered.
        T = 5.0  # seconds
        n = int(T * fs)  # total number of samples
        t = np.linspace(0, T, n, endpoint=False)
        # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
        data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(
            9 * 2 * np.pi * t) + 0.5 * np.sin(12.0 * 2 * np.pi * t)

        # Filter the data, and plot both the original and filtered signals.
        y = self.butter_bandpass_filter(data, cutoff, fs, order)

        plt.subplot(2, 1, 2)
        plt.plot(t, data, 'b-', label='data')
        plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

        plt.subplots_adjust(hspace=0.35)

    def apply_bandpass_filtering(self, data, low_cutoff=80, high_cutoff=1000,
                                 order=4, fs=16000, duration=1.0,
                                 is_plot=False):
        y = self.butter_bandpass_filter(data, low_cutoff, high_cutoff, fs,
                                        order)

        if is_plot:
            samples = int(fs * duration)
            t = np.arange(samples) / fs
            plt.plot(t, data, 'b-', label='data')
            plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
            plt.xlabel('Time [sec]')
            plt.grid()
            plt.legend()

            plt.subplots_adjust(hspace=0.35)
        return y

    @staticmethod
    def autocorrelation(x):
        """
        Compute the autocorrelation of the signal, based on the properties of the
        power spectral density of the signal.
        """
        xp = x - np.mean(x)
        f = np.fft.fft(xp)
        p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
        pi = np.fft.ifft(p)
        a = np.real(pi)[:int(x.size / 2)]
        s = np.sum(xp ** 2)
        if s > 0:
            return a / s
        else:
            return np.zeros(shape=a.shape)

    @staticmethod
    def get_zero_crossing(data):
        idx_zc = np.where(np.diff(np.sign(data)))[0]
        return idx_zc

    @staticmethod
    def apply_threshold_greater(x, threshold):
        return x[np.where(x > threshold)]

    @staticmethod
    def apply_threshold_smaller(x, threshold):
        return x[np.where(x < threshold)]

    @staticmethod
    def running_mean(x, n):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    def sd_preprocess(self, fname, is_plot):
        signal = self._read_wav_and_pad(fname)
        return signal

    def sd_amplitude_envelop(self, signal, threshold, is_plot):
        amplitude_envelop = self.get_amplitude_envelop(signal, is_plot=is_plot)
        amplitude_envelop2 = amplitude_envelop - np.mean(amplitude_envelop)
        idx_thres_ae = np.where(amplitude_envelop > threshold)[0]
        # plt.plot(amplitude_envelop)
        # plt.plot(amplitude_envelop2)
        # normalized_ae = normalize(amplitude_envelop, axis=0).ravel()
        # filtered_ae = np.zeros(shape=normalized_ae.shape)
        # filtered_ae[idx_thres_ae] = 1
        # plt.plot(normalized_ae)
        # plt.plot(filtered_ae)
        original_len = len(signal)
        filtered_len = len(idx_thres_ae)
        ratio = filtered_len / float(original_len)
        logging.log(logging.DEBUG, "ratio_ae:"+str(ratio))
        if ratio > 0.5:
            return 1
        else:
            return 0

    def sd_autocorrelation(self, signal, threshold):
        # signal_acorr = self.autocorrelation(signal)
        # signal_acorr = self.apply_threshold(np.abs(signal_acorr[1:]),
        # threshold)
        acorr = acf(signal, nlags=1, fft=True)
        logging.log(logging.DEBUG, "acorr:" + str(acorr[1]))
        if acorr[1] > threshold:
            return 1
        else:
            return 0

    def sd_zero_crossing(self, signal, threshold):
        idx_zc = self.get_zero_crossing(signal)
        ratio_zc = len(idx_zc) / float(len(signal))
        logging.log(logging.DEBUG, "ratio_zc:" + str(ratio_zc))
        if ratio_zc < threshold:
            return 1
        else:
            return 0

    def silence_detection(self, fname, threshold_db=3.0,
                          threshold_acorr=0.3, threshold_zero_crossing=0.3,
                          is_plot=False):
        signal = self.sd_preprocess(fname, is_plot)
        # f, Pxx_spec = welch(signal, 16000, 'flattop', 1024,
        # scaling='spectrum')
        # plt.figure()
        # plt.semilogy(f, np.sqrt(Pxx_spec))
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('Linear spectrum [V RMS]')
        # plt.title(os.path.basename(fname))
        # n = int(16000 * 0.2)
        # print(self.running_mean(signal, n))

        w_amplitude = 0.33
        w_acorr = 0.33
        w_zero_crossing = 0.33
        sg_amplitude = self.sd_amplitude_envelop(signal, threshold_db, is_plot)
        sg_acorr = self.sd_autocorrelation(signal, threshold_acorr)
        sg_zero_crossing = self.sd_zero_crossing(signal,
                                                 threshold_zero_crossing)
        sg = list()
        sg.append(sg_amplitude)
        sg.append(sg_acorr)
        sg.append(sg_zero_crossing)
        is_speech = np.prod(sg)
        logging.log(logging.DEBUG, "result:" + ",".join([str(x) for x in sg]))
        if is_plot:
            plt.show()
        silence = 1 - is_speech
        return silence

    def testcase(self):
        ps_fname = "/Users/maureen/Documents/Work/kaggle/assets/data_augmentation/silence/pure_silence.wav"
        bn_pathnames = "/Users/maureen/Documents/Work/kaggle/assets/train/audio/_background_noise_/*.wav"
        dg_pathnames = "/Users/maureen/Documents/Work/kaggle/assets/train" \
                       "/audio/dog/*.wav"
        test_sil_pathnames = "/Users/maureen/Documents/Work/kaggle_additional_data/label/silence/*.wav"
        bn_fnames = glob.glob(bn_pathnames)
        dg_fnames = glob.glob(dg_pathnames)
        ts_fnames = glob.glob(test_sil_pathnames)
        thres_db = 3
        thres_acorr = 0.3
        thres_zero_crossing = 0.3
        # for fname in ts_fnames:
        #
        #     result = self.silence_detection(fname, threshold_db=thres_db,
        #                                     threshold_acorr=thres_acorr,
        #                                     threshold_zero_crossing=thres_zero_crossing)
        #     #print(os.path.basename(fname), result)
        #     logging.log(logging.DEBUG, "File:" + os.path.basename(fname) +
        #                 "--" + str(result))

        acc = 0
        for fname in dg_fnames:
            result = self.silence_detection(fname,
                                            threshold_db=thres_db,
                                            threshold_acorr=thres_acorr,
                                            threshold_zero_crossing=thres_zero_crossing)
            logging.log(logging.DEBUG, "File:" + os.path.basename(fname) +
                        "--" + str(result))
            acc += result
        print(1-(acc / len(dg_fnames)))
        plt.show()


if __name__ == "__main__":
    SC = SilenceDetector()
    SC.testcase()
    # SC.demo_lf()
