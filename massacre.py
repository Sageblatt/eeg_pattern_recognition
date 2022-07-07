from math import floor
import numpy as np
import pandas as pd
from EDFlib import edfreader
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------#
ADDITIONAL_DATA = False

fname = 'data/28-05-2016_19-00_reduced_IIS.bdf'
fnum = 3                                               # the number of file in the archive
hdl = edfreader.EDFreader(fname)

edfsignals = hdl.getNumSignals()
samples_amount = np.zeros(edfsignals, dtype = np.int32)
sample_freq = np.zeros(edfsignals)

print("Number of signals in file: %d" % (hdl.getNumSignals()))

print("\nSignal list:")
for i in range(edfsignals):
    print("\nSig. %d: %s" % (i, hdl.getSignalLabel(i)))
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

print("Select signal (enter value from 0 to %d):" % (edfsignals - 1))
sig_num = int(input())

sig = np.empty(samples_amount[sig_num], dtype = np.int32)

hdl.rewind(0)
hdl.readSamples(sig_num, sig, samples_amount[sig_num])
hdl.close()

t = np.linspace(0, samples_amount[sig_num] / sample_freq[sig_num], samples_amount[sig_num])

#------------------------------------------------------------------------------------------#

data = pd.read_excel('data/Spikes.xlsx', index_col = 0)
spikes = []
spikes.append(data.loc[:, "latency"].to_numpy())
spikes.append(data.loc[:, "duration"].to_numpy())
spikes = np.array(spikes)

sample_len = floor(np.max(spikes[1]))
print(sample_len)

res = []

for i in range(len(spikes[0])):
    spike = []
    time = []
    spike_len = floor(spikes[1][i])
    spike_start = floor(spikes[0][i])
    start =  spike_start - floor((sample_len - spike_len) / 2)
    end = spike_start + spike_len + floor((51 - spike_len) / 2)
    if (end-start) != sample_len:
        end += sample_len - (end - start)
    for n in range(start, end):
        spike.append(sig[n])
        time.append(t[n])
    print('num of spike: ', i, ' spike len: ', spike_len, ' latency of sample: ', end-start)
    plt.plot(time, spike, figure=plt.figure(figsize=(10.0, 6.0)))
    plt.show()
    res.append(spike)
    
res = np.array(res, dtype=object)
res = np.asanyarray(res)
#plt.plot(t, sig, figure=plt.figure(figsize=(10.0, 6.0)))
#plt.show()
pd.DataFrame(res).to_csv('spikes/' + str(fnum) + '_' + str(sig_num) + '.csv')
