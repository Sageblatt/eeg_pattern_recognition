import numpy as np
from EDFlib import edfreader


ADDITIONAL_DATA = False

fname = '27-02-2015_19-49_reduced 40 sec.bdf'

hdl = edfreader.EDFreader(fname)


edfsignals = hdl.getNumSignals()
samples_amount = np.zeros(edfsignals, dtype=np.int32)
sample_freq = np.zeros(edfsignals)
print("Number of signals in file: %d" % (hdl.getNumSignals()))
print("Recording duration: %f s" % (hdl.getLongDataRecordDuration() / 10000000.0 * hdl.getNumDataRecords()))

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
# n = int(input())
n = 0

sig = np.empty(samples_amount[n], dtype = np.int32)
# dbuf = np.empty(100, dtype = np.float_)

hdl.rewind(0)
hdl.readSamples(n, sig, samples_amount[n])
# hdl.rewind(0)
# hdl.readSamples(0, dbuf, 100)
hdl.close()

t = np.linspace(0, samples_amount[n] / sample_freq[n], samples_amount[n])


import matplotlib.pyplot as plt

plt.plot(t, sig, figure=plt.figure(figsize=(10.0, 6.0)))
plt.show()
