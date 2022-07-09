import numpy as np
import scipy.signal as sp
import pandas as pd
from statistics import mean 
from math import floor
from EDFlib import edfreader
import matplotlib.pyplot as plt

ADDITIONAL_DATA = False

fname = '27-02-2015_19-49_reduced 300 sec.bdf'

hdl = edfreader.EDFreader(fname)

edfsignals = hdl.getNumSignals()
samples_amount = np.zeros(edfsignals, dtype=np.int32)
sample_freq = np.zeros(edfsignals)

print("Number of signals in file: %d" % (hdl.getNumSignals()))
print("Recording duration: %f s" % (hdl.getLongDataRecordDuration() / 10000000.0 * hdl.getNumDataRecords()))

print("\nSignal list:")

for i in range(edfsignals):
    #print("\nSig. %d: %s" % (i, hdl.getSignalLabel(i)))
    samples_amount[i] = hdl.getTotalSamples(i)
    sample_freq[i] = hdl.getSampleFrequency(i)
    if ADDITIONAL_DATA:
        print("Samplefrequency: %f Hz" % (hdl.getSampleFrequency(i)))
        print("Physical dimension: %s" % (hdl.getPhysicalDimension(i)))
        print("Physical minimum: %f" % (hdl.getPhysicalMinimum(i)))
        print("Physical maximum: %f" % (hdl.getPhysicalMaximum(i)))
        print("Digital minimum: %d" % (hdl.getDigitalMinimum(i)))
        print("Digital maximum: %d" % (hdl.getDigitalMaximum(i)))
        print("Total samples in file: %d" % (hdl.getTotalSamples(i)))

#print("Select signal (enter value from 0 to %d):" % (edfsignals - 1))
n = int(input())
#n = 1

sig = np.empty(samples_amount[n], dtype = np.int32)
# dbuf = np.empty(100, dtype = np.float_)

hdl.rewind(0) 
hdl.readSamples(n, sig, samples_amount[n])
# hdl.rewind(0)
# hdl.readSamples(0, dbuf, 100)
hdl.close()

t = np.linspace(0, samples_amount[n] / sample_freq[n], samples_amount[n])

plt.plot(t, sig, figure=plt.figure(figsize=(10.0, 6.0)))
#plt.plot(t[250:-2], (np.diff(np.diff(sig))/np.diff(np.diff(t)))[250:])
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------------------

high_peaks_indexes = sp.find_peaks(sig)[0]
high_peaks = [sig[i] for i in high_peaks_indexes]
low_peaks_indexes = sp.find_peaks(sig*-1)[0]
low_peaks = [sig[i] for i in low_peaks_indexes]

if(len(high_peaks) < len(low_peaks)):
    delta = [high_peaks[i] - low_peaks[i] for i in range(len(high_peaks))]
else:
    delta = [high_peaks[i] - low_peaks[i] for i in range(len(low_peaks))]

wigth_avg = mean(delta)

spikes_num = 0
spikes = []
times = []
while i < len(high_peaks) - 2:
    if(abs(high_peaks[i] - high_peaks[i+2]) >= 0.35*wigth_avg or abs(high_peaks[i] - high_peaks[i+1]) >= 0.3*wigth_avg):
        spikes_num = spikes_num + 1
        spike = []
        time = []
        for j in sig[high_peaks_indexes[i - 2]: high_peaks_indexes[i+5]]:
            spike.append(j)
        for j in t[high_peaks_indexes[i - 2]: high_peaks_indexes[i+5]]:
            time.append(j)
        spikes.append(spike)
        times.append(time)
        i = i + 4
    else:
        i = i + 1
print(spikes_num)
    
for i in range(len(spikes)):

    plt.plot(times[i], spikes[i])
    plt.show()
    

