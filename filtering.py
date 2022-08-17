import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spf
from scipy import signal
from os import path

# This script filters low frequencies from raw files,
# removes noises and normalizes data to (-1, 1) range

def cut_low(s, fs):  
    # Cuts every frequency below freq, cuts signal if it can't be floor divided by 10
    def high_pass(signal, freq, sample_frequency):
        m = np.size(signal)
        x_f = spf.rfftfreq(m, 1 / sample_frequency)
        y_f = spf.rfft(signal)
        y_f[(x_f <= freq)] = 0
        return spf.irfft(y_f), y_f
    
    # Optional choice for slicing signal before filtering
    # piece = fs
    # if t_stop % 4 == 0:
    #     piece = sample_freq[n] * 4
    # elif t_stop % 3 == 0:
    #     piece = sample_freq[n] * 3
    # elif t_stop % 2 == 0:
    #     piece = sample_freq[n] * 2
    
    piece = int(10)
    while len(s) % piece != 0:
        s = s[:-1]
    cut_sig = np.array_split(s, len(s) / piece)

    for i in range(len(cut_sig)):
        cut_sig[i], _ = high_pass(cut_sig[i], 1, fs)

    sig_filt = np.concatenate(cut_sig)
    yf_filt = spf.rfft(sig_filt)
    return sig_filt, yf_filt


def analyze(s, fs):
    xf = spf.rfftfreq(len(s), 1 / fs)
    yf = np.abs(spf.rfft(s))
    
    freqs = [25, 50, 75, 100, 125]
    coefs = [5, 3, 5, 3, 5]
    bool_freqs = np.zeros(5)
    
    for i in range(len(freqs)):
        if np.max(yf[np.abs(xf - freqs[i]) < 0.3]) > \
            coefs[i] * np.mean(yf[np.logical_and(np.abs(xf - freqs[i]) > 0.5, np.abs(xf - freqs[i]) < 1.5)]):
            bool_freqs[i] = 1
    return bool_freqs


def denoise(s, frs, filters):
    sos = []
    sig_ = np.copy(s)
    
    if filters[0]:
        ord, wn = signal.buttord([24.5, 26.5], [24.9, 25.1], 2, 5, fs=frs)
        sos.append(signal.butter(ord, wn, 'bs', fs=frs, output='sos'))
    if filters[1]:
        ord, wn = signal.buttord([49, 51], [49.9, 50.1], 5, 30, fs=frs)
        sos.append(signal.butter(ord, wn, 'bs', fs=frs, output='sos'))
    if filters[2]:
        ord, wn = signal.buttord([74.5, 76.5], [74.9, 75.1], 2, 5, fs=frs)
        sos.append(signal.butter(ord, wn, 'bs', fs=frs, output='sos'))
    if filters[3]:
        ord, wn = signal.buttord([99, 101], [99.95, 100.05], 3, 30, fs=frs)
        sos.append(signal.butter(ord, wn, 'bs', fs=frs, output='sos'))
    if filters[4]:
        # 125 Hz works only for 250 Hz sample frequency!
        if frs > 252:
            raise ValueError('Sampling frequency is higher than 250 Hz. Unsupported file.')
        ord, wn = signal.buttord(124.7, 124.8, 15, 25, fs=frs)
        sos.append(signal.butter(ord, wn, 'lp', fs=frs, output='sos'))
    
    for f in sos:
        sig_ = signal.sosfilt(f, sig_)
        
    yf_ = spf.rfft(sig_)
    return sig_, yf_

def normalize(s):
    up = np.max(s)
    down = np.min(s)
    b = (up + down) / (up - down)
    a = (1 - b) / down
    return a * s + b


if __name__ == '__main__':
    # save filtered signal as .npy to ./data/np_filt folder
    SAVE_RESULT = 0

    # Enables matplotlib plot of signal
    PREVIEW = 1
    
    names = ['40',
             '300',
             'IIS',
             'big']

    fnum = 2
    channel = 6
    
    dir_path = path.dirname(path.realpath(__file__))
    fname = path.join(dir_path, "data", "np_raw", names[fnum] + ' Channel ' + str(channel) + '       .npy')

    sig_mod = np.load(fname)
    del fname
    
    fs = sig_mod[0]
    sig = sig_mod[1:]
    

    sig_filt, yf_filt = cut_low(sig, fs)
    bool_filters = analyze(sig_filt, fs)
    sig_den, yf_den = denoise(sig_filt, fs, bool_filters)
    sig_filt, yf_filt = cut_low(sig_den, fs)

    if PREVIEW:
        samples_amount = len(sig)
        t = np.linspace(0, samples_amount / fs, samples_amount)
        xf = spf.rfftfreq(samples_amount, 1 / fs)
        yf = spf.rfft(sig)
        
        t1 = np.linspace(0, len(sig_den) / fs, len(sig_den))
        xf1 = spf.rfftfreq(len(sig_den), 1 / fs)
        
        t2 = np.linspace(0, len(sig_filt) / fs, len(sig_filt))
        xf2 = spf.rfftfreq(len(sig_filt), 1 / fs)
        
        fig, axes = plt.subplots(ncols=3, nrows=2, gridspec_kw={"wspace": 0.2, "hspace": 0.5}, figsize=[14.0, 7.0])

        axes[0, 0].plot(t, sig)
        axes[1, 0].plot(xf, np.abs(yf))
        axes[0, 1].plot(t1, sig_den)
        axes[1, 1].plot(xf1, np.abs(yf_den))
        axes[0, 2].plot(t2, sig_filt)
        axes[1, 2].plot(xf2, np.abs(yf_filt))

        plt.show()

    # first element of array is sampling frequency
    if SAVE_RESULT:
        sig_mod = np.concatenate([[fs], sig_den])
        fname = path.join(dir_path, "data", "np_filt", names[fnum] + ' Channel_' + str(channel) + '_filt')
        np.save(fname, sig_mod)
    