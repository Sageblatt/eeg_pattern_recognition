import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as spf
from scipy import signal

SAVE_RESULT = True

names = ['27-02-2015_19-49_reduced 40 sec',
         '27-02-2015_19-49_reduced 300 sec',
         '28-05-2016_19-00_reduced_IIS']

fnum = 0
channel = 1
fname = 'data/np_raw/' + names[fnum] + ' Channel ' + str(channel) + '       .npy'

sig_mod = np.load(fname)
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

# ------------------ Filtering 50 Hz noise ----------------- #
ord, wn = signal.buttord([49, 51], [49.9, 50.1], 5, 50, fs=fs)
print(ord, wn)
sos = signal.butter(ord, wn, 'bs', fs=fs, output='sos')

ord, wn = signal.buttord([99, 101], [99.95, 100.05], 3, 30, fs=fs)
print(ord, wn)
sos1 = signal.butter(ord, wn, 'bs', fs=fs, output='sos')
# sos = signal.butter(1, (49.5, 50.52), 'bs', fs=fs, output='sos')

sig_den = signal.sosfilt(sos, sig_filt)
sig_den = signal.sosfilt(sos1, sig_den)
yf_den = spf.rfft(sig_den)
# ---------------------------------------------------------- #

test = 0
if test:
    yf_filt[:] = 10
    sig_filt = spf.irfft(yf_filt)
    sig_den = signal.sosfilt(sos, sig_filt)
    yf_den = spf.rfft(sig_den)


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
    np.save('data/np_filt/' + names[fnum] + ' Channel_' + str(channel) + '_filt', sig_mod)

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