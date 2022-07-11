import numpy as np
from EDFlib import edfreader
import matplotlib.pyplot as plt


# open signal as plot in mpl; incompatible with save_all mode
PREVIEW = False

# show available data for each channel
ADDITIONAL_DATA = False

# save signal as numpy array to data/np_raw folder (first value in array is sample frequency, then comes raw signal)
SAVE_TO_NUMPY = True
# save every channel in file or one chosen
SAVE_ALL = True

# choose file
names = ['27-02-2015_19-49_reduced 40 sec',
         '27-02-2015_19-49_reduced 300 sec',
         '28-05-2016_19-00_reduced_IIS']

fnum = 2
fname = 'data/raw/' + names[fnum] + '.bdf'

hdl = edfreader.EDFreader(fname)

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
        np.save('data/np_raw/' + names[fnum] + ' ' + channel_names[n], sig_mod)
else:
    for i in range(edfsignals):
        sig = np.empty(samples_amount[i], dtype=np.int32)

        hdl.rewind(0)
        hdl.readSamples(i, sig, samples_amount[i])

        sig_mod = np.concatenate([[sample_freq[i]], sig])
        np.save('data/np_raw/' + names[fnum] + ' ' + channel_names[i], sig_mod)
    hdl.close()
