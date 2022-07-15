import numpy as np
import scipy.signal as sp
import pandas as pd
from statistics import mean 
from math import floor
from EDFlib import edfreader
import matplotlib.pyplot as plt
import scipy.fft as spf
#from scipy import signal

ADDITIONAL_DATA = False

fname = 'C:\\Users\\Anton\\Downloads\\Spikes_sample\\data\\data\\np_filt\\27-02-2015_19-49_reduced 300 sec Channel_2_filt.npy'
sig_mod = np.load(fname)
fs = sig_mod[0]
sig = sig_mod[1: ] 

samples_amount = len(sig)
t_stop = samples_amount / fs
t = np.linspace(0, t_stop, samples_amount)

plt.plot(t, sig, figure=plt.figure(figsize=(10.0, 6.0)))
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------

high_peaks_indexes = sp.find_peaks(sig)[0]
high_peaks = [sig[i] for i in high_peaks_indexes]
low_peaks_indexes = sp.find_peaks(sig*-1)[0]
low_peaks = [sig[i] for i in low_peaks_indexes]

delta = []
if(len(high_peaks) < len(low_peaks)):
    for i in range(len(high_peaks)):
        if(high_peaks[i] - low_peaks[i] > 0.05):
            delta.append(high_peaks[i] - low_peaks[i])            
else:
    for i in range(len(low_peaks)):
        if(high_peaks[i] - low_peaks[i] > 0.05):
            delta.append(high_peaks[i] - low_peaks[i])
            
'''
if(len(high_peaks) < len(low_peaks)):
    delta = [high_peaks[i] - low_peaks[i] for i in range(len(high_peaks))]
else:
    delta = [high_peaks[i] - low_peaks[i] for i in range(len(low_peaks))]
ddelta = []
for i in delta:
    if(i >= wigth_avg):
        ddelta.append(i)
wwigth_avg = mean(ddelta)
'''

wigth_avg = mean(delta)
print(wigth_avg)

spikes_num = 0
spikes = []
times = []
i = int(0)
while i < len(high_peaks) - 2:
    if(abs(high_peaks[i] - low_peaks[i]) >= 4*wigth_avg or abs(high_peaks[i] - low_peaks[i+1]) >= 4*wigth_avg):
        spikes_num = spikes_num + 1
        spike = []
        time = []
        if(i >= 2 and (i + 5) <= len(high_peaks) - 1):
            for j in sig[high_peaks_indexes[i - 2]: high_peaks_indexes[i+5]]:
                spike.append(j)
            for k in t[high_peaks_indexes[i - 2]: high_peaks_indexes[i+5]]:
                time.append(k)
        elif (i < 2 or (i + 5) > len(high_peaks) - 1):
            if(i < 2):
                delt1 = 0
            else:
                delt1 = i - 2
            if(i + 5 > len(high_peaks) - 1):
                delt2 = len(high_peaks) - i - 1
            else:
                delt2 = 4
            for j in sig[high_peaks_indexes[delt1]: high_peaks_indexes[i+delt2]]:
                spike.append(j)
            for k in t[high_peaks_indexes[delt1]: high_peaks_indexes[i+delt2]]:
                time.append(k)
        spikes.append(spike)
        times.append(time)
        i = i + 4
    else:
        i = i + 1
print(spikes_num)


for i in range(len(times)):
    plt.plot(times[i], spikes[i])
    plt.show()
'''
for i in range(len(times)):
    print(times[i][0], spikes[i][0])
'''
