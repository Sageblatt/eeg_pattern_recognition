import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spf
from scipy import signal
from os import path

# This script filters low frequencies from raw files,
# removes noises and normalizes data to (-1, 1) range

# save filtered signal as .npy to ./data/np_filt folder
SAVE_RESULT = True

# Cut 25, 75 and 125 Hz or not
ADDITIONAL_FREQS = False

# Cut 100 Hz
SECOND_HARM = True

# Enables matplotlib plot of signal
PREVIEW = False

names = ['40',
         '300',
         'IIS']

fnum = 2
channel = 6
dir_path = path.dirname(path.realpath(__file__))
fname = path.join(dir_path, "data", "np_raw", names[fnum] + ' Channel ' + str(channel) + '       .npy')

sig_mod = np.load(fname)
del fname
fs = sig_mod[0]
sig = sig_mod[1:]
samples_amount = len(sig)

t_stop = samples_amount / fs
t = np.linspace(0, t_stop, samples_amount)

# ------------------ Cutting low frequencies ----------------- #
# Optional choice for slicing signal before filtering
# piece = fs
# if t_stop % 4 == 0:
#     piece = sample_freq[n] * 4
# elif t_stop % 3 == 0:
#     piece = sample_freq[n] * 3
# elif t_stop % 2 == 0:
#     piece = sample_freq[n] * 2

piece = int(10)

# Cuts every frequency below freq
def high_pass(signal, freq, sample_frequency):
    m = np.size(signal)
    x_f = spf.rfftfreq(m, 1 / sample_frequency)
    y_f = spf.rfft(signal)
    y_f[(x_f <= freq)] = 0
    return spf.irfft(y_f), y_f


cut_sig = np.array_split(sig, samples_amount / piece)

for i in range(len(cut_sig)):
    cut_sig[i], _ = high_pass(cut_sig[i], 1, fs)

sig_filt = np.concatenate(cut_sig)

xf = spf.rfftfreq(samples_amount, 1 / fs)
yf = spf.rfft(sig)
yf_filt = spf.rfft(sig_filt)
# ---------------------------------------------------------- #

# ------------------ Filtering noise ----------------- #
sig_den = sig_filt
sos = []

# 50 Hz
ord, wn = signal.buttord([49, 51], [49.9, 50.1], 5, 30, fs=fs)
sos.append(signal.butter(ord, wn, 'bs', fs=fs, output='sos'))

# 100 Hz
if SECOND_HARM:
    ord, wn = signal.buttord([99, 101], [99.95, 100.05], 3, 30, fs=fs)
    sos.append(signal.butter(ord, wn, 'bs', fs=fs, output='sos'))

# 25, 75, 125 Hz
if ADDITIONAL_FREQS:
    ord, wn = signal.buttord([24.5, 26.5], [24.9, 25.1], 2, 5, fs=fs)
    sos.append(signal.butter(ord, wn, 'bs', fs=fs, output='sos'))
    ord, wn = signal.buttord([74.5, 76.5], [74.9, 75.1], 2, 5, fs=fs)
    sos.append(signal.butter(ord, wn, 'bs', fs=fs, output='sos'))
    # 125 Hz works only for 250 Hz sample frequency!
    ord, wn = signal.buttord(124.7, 124.8, 15, 25, fs=fs)
    sos.append(signal.butter(ord, wn, 'lp', fs=fs, output='sos'))

for s in sos:
    sig_den = signal.sosfilt(s, sig_den)


yf_den = spf.rfft(sig_den)
# ---------------------------------------------------------- #

# -------------------Normalizing---------------------------- #
up = np.max(sig_den)
down = np.min(sig_den)
b = (up + down) / (up - down)
a = (1 - b) / down
sig_den = a * sig_den + b
# ---------------------------------------------------------- #

# test = 0
# if test:
#     yf_filt[:] = 10
#     sig_filt = spf.irfft(yf_filt)
#     sig_den = signal.sosfilt(sos, sig_filt)
#     yf_den = spf.rfft(sig_den)

if PREVIEW:
    fig, axes = plt.subplots(ncols=3, nrows=2, gridspec_kw={"wspace": 0.2, "hspace": 0.5}, figsize=[14.0, 7.0])

    axes[0, 0].plot(t, sig)
    axes[1, 0].plot(xf, np.abs(yf))
    axes[0, 1].plot(t, sig_filt)
    axes[1, 1].plot(xf, np.abs(yf_filt))
    axes[0, 2].plot(t, sig_den)
    axes[1, 2].plot(xf, np.abs(yf_den))


    plt.show()

# first element of array is sampling frequency
if SAVE_RESULT:
    sig_mod = np.concatenate([[fs], sig_den])
    fname = path.join(dir_path, "data", "np_filt", names[fnum] + ' Channel_' + str(channel) + '_filt')
    np.save(fname, sig_mod)

# Test filter's frequency response for de-noise
# b, a = signal.butter(2, (49.8, 50.22), 'bandstop', analog=True)
# b, a = signal.butter(ord, wn, 'bandstop', analog=True)
# w, h = signal.freqs(b, a)
# plt.semilogx(w, 20 * np_raw.log10(abs(h)))
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')
# plt.axvline(100, color='green') # cutoff frequency