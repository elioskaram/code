import numpy as np
import biosppy.signals.ecg as ecg
import scipy.signal as signal
from scipy.signal import butter, filtfilt


def get_ecg_mean(ecg_signal):
    return np.mean(ecg_signal)

def get_ecg_std(ecg_signal):
    return np.std(ecg_signal)

def get_ecg_rms(ecg_signal):
    return np.sqrt(np.mean(ecg_signal**2))

def get_ecg_heart_rate(ecg_signal, sampling = 4000):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling, show=False)
    return out['heart_rate'].mean()

def get_ecg_sdnn(ecg_signal, sampling = 4000):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling, show=False)
    nn_intervals = np.diff(out['rpeaks'])
    sdnn = np.std(nn_intervals)
    return sdnn

def get_ecg_rmssd(ecg_signal, sampling = 4000):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling, show=False)
    nn_intervals = np.diff(out['rpeaks'])
    successive_diff = np.diff(nn_intervals)
    return np.sqrt(np.mean(successive_diff**2))

def get_ecg_pnn50(ecg_signal, sampling = 4000):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling, show=False)
    nn_intervals = out['rpeaks']
    nn50_count = np.sum(np.abs(nn_intervals) > 50)
    pnn50 = (nn50_count / len(nn_intervals)) * 100
    return pnn50

def butterworth_lowpass_filter(ecg_signal, cutoff_freq, sampling_rate = 4000):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = signal.butter(N=4, Wn=normal_cutoff_freq, btype='low')
    filtered_signal = signal.filtfilt(b, a, ecg_signal, method='gust')
    return 3 * filtered_signal