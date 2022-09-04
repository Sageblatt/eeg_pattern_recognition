import numpy as np
import scipy.signal as sp
import pandas as pd


def candidates(sig, freq):
    samples_amount = len(sig)
    t_stop = samples_amount / freq
    t = np.linspace(0, t_stop, samples_amount)
    
    high_peaks_indexes = sp.find_peaks(sig)[0]
    high_peaks = [sig[i] for i in high_peaks_indexes]
    low_peaks_indexes = sp.find_peaks(sig*-1)[0]
    low_peaks = [sig[i] for i in low_peaks_indexes]

    delta = []
    if len(high_peaks) < len(low_peaks):
        for i in range(len(high_peaks)):
            if high_peaks[i] - low_peaks[i] > 0.05:
                delta.append(high_peaks[i] - low_peaks[i])            
    else:
        for i in range(len(low_peaks)):
            if high_peaks[i] - low_peaks[i] > 0.05:
                delta.append(high_peaks[i] - low_peaks[i])
                
    width_avg = np.mean(delta)
    spikes_num = 0
    spikes = []
    times = []
    i = 0
    while i < len(high_peaks) - 2:
        if abs(high_peaks[i] - low_peaks[i]) >= 4 * width_avg or abs(high_peaks[i] - low_peaks[i+1]) >= 4 * width_avg:
            spikes_num = spikes_num + 1
            spike = []
            time = []
            
            if i >= 2 and i + 5 <= len(high_peaks) - 1:
                for j in sig[high_peaks_indexes[i-2]:high_peaks_indexes[i+5]]:
                    spike.append(j)
                for k in t[high_peaks_indexes[i-2]:high_peaks_indexes[i+5]]:
                    time.append(k)
            elif i < 2 or i + 5 > len(high_peaks) - 1:
                if i < 2:
                    delt1 = 0
                else:
                    delt1 = i - 2
                if i + 5 > len(high_peaks) - 1:
                    delt2 = len(high_peaks) - i - 1
                else:
                    delt2 = 4
                for j in sig[high_peaks_indexes[delt1]:high_peaks_indexes[i+delt2]]:
                    spike.append(j)
                for k in t[high_peaks_indexes[delt1]:high_peaks_indexes[i+delt2]]:
                    time.append(k)
            spikes.append(spike)
            times.append(time)
            i = i + 4
        else:
            i = i + 1
    
    duration = np.array([len(spikes[i]) for i in range(len(spikes))])
    latency = np.array([times[i][0] for i in range(len(times))])

    res = {'duration': duration, 'latency': latency}
    return res

if __name__ == '__main__':
    fnum = 1
    fname = 'data\\np_filt\\27-02-2015_19-49_reduced 300 sec Channel_2_filt.npy'
    sig_mod = np.load(fname)
    fs = sig_mod[0]
    sig = sig_mod[1: ]
    res1 = pd.DataFrame(candidates(sig, fs))
    writer = pd.ExcelWriter('spikes' + str(fnum) + '.xlsx')
    res1.to_excel(writer)
    writer.save()
    '''
    for i in range(len(times)):
        plt.plot(times[i], spikes[i])                            #spikes on graphs
        plt.show()
    for i in range(len(times)):         
        print(str(len(spikes[i])) + '  ' + str(times[i][0]))     #latency and duration of spikes, printed
    '''


