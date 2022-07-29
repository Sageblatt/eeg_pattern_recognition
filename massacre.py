from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for sig_num in range(1, 7):

    sig = np.load('data/np_filt/IIS Channel_' + str(sig_num) + '_filt.npy')
    t = np.linspace(0, len(sig) / 250.42798142, len(sig))

    data = pd.read_excel('data/Spikes.xlsx', index_col = 0)
    spikes = []
    spikes.append(data.loc[:, "Latency_corrected (sec)"].to_numpy())
    spikes.append(data.loc[:, "duration"].to_numpy())
    spikes = np.array(spikes)

    sample_len = floor(np.max(spikes[1]))

    res = []
    not_spikes = []
    starts = []
    ends = []

    for i in range(len(spikes[0])):
        spike = []
        time = []
        
        spike_len = floor(spikes[1][i])
        spike_start = (np.abs(t - spikes[0][i])).argmin()
        
        start =  spike_start - floor((sample_len - spike_len) / 2)
        end = spike_start + spike_len + floor((51 - spike_len) / 2)
        
        starts.append(start)
        ends.append(end)
        
        if (end-start) != sample_len:
            end += sample_len - (end - start)
        for n in range(start, end):
            spike.append(sig[n])
            time.append(t[n])
        
        #plt.plot(time, spike, figure=plt.figure(figsize=(10.0, 6.0)))
        #plt.show()
        
        up = np.max(spike)
        down = np.min(spike)
        b = (up + down) / (up - down)
        a = (1 - b) / down
        spike = a * np.array(spike) + b
        
        #plt.plot(time, spike, figure=plt.figure(figsize=(10.0, 6.0)))
        #plt.show()
        
        res.append(spike)

    res = np.array(res, dtype=object)
    res = np.asanyarray(res)

    pd.DataFrame(res).to_csv('data/spikes.csv', mode='a', index = False, header = False)

    ends = [0] + ends

    for i in range(len(starts)-1):
       if (starts[i] - ends[i]) >= 50:
            t = 0
            while ((starts[i] - (ends[i] + 50 * t)) >= 50):
                not_spike = []
                for k in range(ends[i] + 50 * t, ends[i] + 50 * (t+1) + 1):
                    not_spike.append(sig[k])
                
                up = np.max(not_spike)
                down = np.min(not_spike)
                b = (up + down) / (up - down)
                a = (1 - b) / down
                not_spike = a * np.array(not_spike) + b
                
                not_spikes.append(not_spike) 
                t += 1
                
    not_spikes = np.array(not_spikes, dtype=object)
    not_spikes = np.asanyarray(not_spikes)
    pd.DataFrame(not_spikes).to_csv('data/not_spikes.csv', mode='a', index = False, header = False)
