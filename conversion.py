import numpy as np
from EDFlib import edfreader
import matplotlib.pyplot as plt
from os import path

# This script converts .bdf files from ./data/raw folder to .npy
# files in ./data/np_raw folder for further interactions

# open signal as plot in mpl; incompatible with save_all mode
PREVIEW = 1

# show available data for each channel
ADDITIONAL_DATA = False

# save signal as numpy array to data/np_raw folder (first value in array is sample frequency, then comes raw signal)
SAVE_TO_NUMPY = 1
# save every channel in file or one chosen
SAVE_ALL = 1

# choose file
names = ['40',
         '300',
         'IIS']

fnum = 2
dir_path = path.dirname(path.realpath(__file__))
fname = path.join(dir_path, "data", "raw", names[fnum] + '.bdf')

hdl = edfreader.EDFreader(fname)
del fname

edfsignals = hdl.getNumSignals()
samples_amount = np.zeros(edfsignals, dtype=np.int32)
sample_freq = np.zeros(edfsignals)
channel_names = []

print("Number of signals in file: %d" % (hdl.getNumSignals()))
print("Recording duration: %f s" % (hdl.getLongDataRecordDuration() / 10000000.0 * hdl.getNumDataRecords()))

print("\nSignal list:")
for i in range(edfsignals):
    print("\nSig. %d: %s" % (i, hdl.getSignalLabel(i)))
    samples_amount[i] = hdl.getTotalSamples(i)
    sample_freq[i] = hdl.getSampleFrequency(i)
    channel_names.append(hdl.getSignalLabel(i))
    if ADDITIONAL_DATA:
        print("Samplefrequency: %f Hz" % (hdl.getSampleFrequency(i)))
        print("Physical dimension: %s" % (hdl.getPhysicalDimension(i)))
        print("Physical minimum: %f" % (hdl.getPhysicalMinimum(i)))
        print("Physical maximum: %f" % (hdl.getPhysicalMaximum(i)))
        print("Digital minimum: %d" % (hdl.getDigitalMinimum(i)))
        print("Digital maximum: %d" % (hdl.getDigitalMaximum(i)))
        print("Total samples in file: %d" % (hdl.getTotalSamples(i)))

if not SAVE_ALL:
    print("Select signal (enter value from 0 to %d):" % (edfsignals - 1))
    n = int(input())

    sig = np.empty(samples_amount[n], dtype=np.int32)

    hdl.rewind(0)
    hdl.readSamples(n, sig, samples_amount[n])
    hdl.close()

    t = np.linspace(0, samples_amount[n] / sample_freq[n], samples_amount[n])

    if PREVIEW:
        plt.plot(t, sig, figure=plt.figure(figsize=(10.0, 6.0)))
        plt.show()

    if SAVE_TO_NUMPY:
        sig_mod = np.concatenate([[sample_freq[n]], sig])
        fname = path.join(dir_path, "data", "np_raw", names[fnum] + ' ' + channel_names[n])
        np.save(fname, sig_mod)
else:
    for i in range(edfsignals):
        sig = np.empty(samples_amount[i], dtype=np.int32)

        hdl.rewind(0)
        hdl.readSamples(i, sig, samples_amount[i])

        sig_mod = np.concatenate([[sample_freq[i]], sig])
        fname = path.join(dir_path, "data", "np_raw", names[fnum] + ' ' + channel_names[i])
        np.save(fname, sig_mod)
    hdl.close()
