import numpy as np
from EDFlib import edfreader
import matplotlib.pyplot as plt
from os import path
from scipy import signal

# This script converts .bdf files from ./data/raw folder to .npy
# files in ./data/np_raw folder for further interactions

def read_bdf(filename, show_output=False, extra_data=False, show_annotations=False, downsample=False, ignore_accel=True):
    hdl = edfreader.EDFreader(filename)
    
    n_signals = hdl.getNumSignals()
    n_samples = np.zeros(n_signals, dtype=np.int32)
    freqs = []
    channel_names = []
    signals = []
    accel = []
    
    if show_output:
        print("Number of signals in file: %d" % (hdl.getNumSignals()))
        print("Recording duration: %f s" % (hdl.getLongDataRecordDuration() / 10000000.0 * hdl.getNumDataRecords()))
        print("\nSignal list:")
    
    for i in range(n_signals):
        if show_output:
            print("\nSig. %d: %s" % (i, hdl.getSignalLabel(i)))
        
        name = hdl.getSignalLabel(i)
        if name.find('Accelerometer') != -1 and ignore_accel:
            accel.append(i)
        if accel.count(i) == 0:
            channel_names.append(name)
            freqs.append(hdl.getSampleFrequency(i))
            
        n_samples[i] = hdl.getTotalSamples(i)
        
        if extra_data and show_output:
            print("Sample frequency: %f Hz" % (hdl.getSampleFrequency(i)))
            print("Physical dimension: %s" % (hdl.getPhysicalDimension(i)))
            print("Physical minimum: %f" % (hdl.getPhysicalMinimum(i)))
            print("Physical maximum: %f" % (hdl.getPhysicalMaximum(i)))
            print("Digital minimum: %d" % (hdl.getDigitalMinimum(i)))
            print("Digital maximum: %d" % (hdl.getDigitalMaximum(i)))
            print("Total samples in file: %d" % (hdl.getTotalSamples(i)))
    
    if show_annotations and show_output:
        n = len(hdl.annotationslist)
        print("\nannotations in file: %d" %(n))
        
        for i in range(0, n):
          print("annotation: onset: %d:%02d:%02.3f    description: %s    duration: %d" %(\
                (hdl.annotationslist[i].onset / 10000000) / 3600, \
                ((hdl.annotationslist[i].onset / 10000000) % 3600) / 60, \
                (hdl.annotationslist[i].onset / 10000000) % 60, \
                hdl.annotationslist[i].description, \
                hdl.annotationslist[i].duration))
    
    for i in range(n_signals):
        if accel.count(i) == 0:
            sig = np.empty(n_samples[i], dtype=np.int32)
            hdl.rewind(0)
            hdl.readSamples(i, sig, n_samples[i])
            signals.append(sig)
    
    hdl.close()
    
    
    if downsample:
        for i in range(len(signals)):
            if not np.isclose(freqs[i], 250, atol=1e-3):
                if freqs[i] < 240:
                    raise RuntimeError(f'Signal {channel_names[i]} has sampling frequency of {freqs[i]:.1f} Hz, minimal supported frequency is 240 Hz.')
                time = len(signals[i]) / freqs[i]
                signals[i] = signal.resample(signals[i], int(time*250))
                freqs[i] = int(time*250)/time
    freqs = np.array(freqs)
    return signals, freqs, channel_names


if __name__ == '__main__':
    # open signal as plot in mpl; incompatible with save_all mode
    PREVIEW = 1
    
    # show available data for each channel
    ADDITIONAL_DATA = 1
    
    # save signal as numpy array to data/np_raw folder (first value in array is sample frequency, then comes raw signal)
    SAVE_TO_NUMPY = 0
    # save every channel in file or one chosen
    SAVE_ALL = 0
    
    # choose file
    names = ['40',
              '300',
              'IIS',
              'annot',
              'big']
    
    fnum = 2
    dir_path = path.dirname(path.realpath(__file__))
    fname = path.join(dir_path, "data", "raw", names[fnum] + '.bdf')
    
    signals, freqs, channel_names = read_bdf(fname, True, ADDITIONAL_DATA, False, True, True)
    
    if not SAVE_ALL:
        print("Select signal (enter value from 0 to %d):" % (len(signals) - 1))
        n = int(input())
    
        t = np.linspace(0, len(signals[n]) / freqs[n], len(signals[n]))
        sig = signals[n]
        

        if PREVIEW:
            plt.plot(t, sig, figure=plt.figure(figsize=(10.0, 6.0)))
    
        if SAVE_TO_NUMPY:
            sig_mod = np.concatenate([[freqs[n]], sig])
            fname = path.join(dir_path, "data", "np_raw", names[fnum] + ' ' + channel_names[n])
            np.save(fname, sig_mod)
    else:
        for i in range(len(signals)):
            if channel_names[i].find('Accelerometer') != -1:
                continue
            sig_mod = np.concatenate([[freqs[i]], signals[i]])
            flname = path.join(dir_path, "data", "np_raw", names[fnum] + ' ' + channel_names[i])
            np.save(flname, sig_mod)
        
