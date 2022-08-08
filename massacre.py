from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# Deletes previously made files 
# to protect data from repeating
DELETE_PREV_FILES = False

if DELETE_PREV_FILES:
    try:
        os.remove('data/not_spikes.csv')
        os.remove('data/spikes.csv')
    except FileNotFoundError:
        print('Files not found, nothing to remove')

SPIKE_IN_THE_MIDDLE    = True
SPIKE_IN_THE_END       = False
SPIKE_IN_THE_BEGINNING = False

def spike_in_the_middle():
        
    spike_len = floor(spikes[1][i])
    spike_start = (np.abs(t - spikes[0][i])).argmin()
        
    start =  spike_start - floor((sample_len - spike_len) / 2)
    end = spike_start + spike_len + floor((51 - spike_len) / 2)
        
    if (end-start) != sample_len:
        end += sample_len - (end - start)
        
    return([start, end])
    
def spike_in_the_end():
    
    spike_len = floor(spikes[1][i])
    spike_start = (np.abs(t - spikes[0][i])).argmin()
    
    start = spike_start - floor((51 - spike_len))
    end = start + 51
    
    if (end-start) != sample_len:
        start -= sample_len - (end - start)

    return([start, end])
    
def spike_in_the_beginning():

    spike_len = floor(spikes[1][i])
    start = (np.abs(t - spikes[0][i])).argmin()
    end = start + spike_len + floor((51 - spike_len))
        
    if (end-start) != sample_len:
        end += sample_len - (end - start)
        
    return([start, end])
    

def normalization(spike):
    up = np.max(spike)
    down = np.min(spike)
    b = (up + down) / (up - down)
    a = (1 - b) / down
    spike = a * np.array(spike) + b
    return(spike)

for sig_num in range(1, 7):
    if sig_num == 5:
        continue

    sig_ = np.load('data/np_filt/IIS Channel_' + str(sig_num) + '_filt.npy')
    fs = sig_[0]
    sig = sig_[1:]
    t = np.linspace(0, len(sig) / fs, len(sig))

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
        
        if (SPIKE_IN_THE_MIDDLE):
            data = spike_in_the_middle()
            start, end = data[0], data[1]
        if (SPIKE_IN_THE_END):
            data = spike_in_the_end()
            start, end = data[0], data[1]
        if (SPIKE_IN_THE_BEGINNING):   
            data = spike_in_the_beginning() 
            start, end = data[0], data[1]
            
        starts.append(start)
        ends.append(end) 
           
        for n in range(start, end):
            spike.append(sig[n])
            time.append(t[n])
        
        #plt.plot(time, spike, figure=plt.figure(figsize=(10.0, 6.0)))
        #plt.show()

        spike = normalization(spike)
        
        #plt.plot(time, spike, figure=plt.figure(figsize=(10.0, 6.0)))
        #plt.show()
        
        res.append(spike)

    res = np.array(res, dtype=object)
    res = np.asanyarray(res)

    pd.DataFrame(res).to_csv('data/spikes.csv', mode='a', index = False, header = False)

    ends = [0] + ends

    for i in range(len(starts)-1):
       if (starts[i] - ends[i]) >= 50:
            p = 0
            while ((starts[i] - (ends[i] + 50 * p)) >= 50):
                not_spike = []
                time = []
                for k in range(ends[i] + 50 * p, ends[i] + 50 * (p+1) + 1):
                    not_spike.append(sig[k])
                    time.append(t[k])
                
                #plt.plot(time, not_spike, figure=plt.figure(figsize=(10.0, 6.0)))
                #plt.show()
                   
                not_spike = normalization(not_spike)
                
                not_spikes.append(not_spike) 
                p += 1
                
    not_spikes = np.array(not_spikes, dtype=object)
    not_spikes = np.asanyarray(not_spikes)
    pd.DataFrame(not_spikes).to_csv('data/not_spikes.csv', mode='a', index = False, header = False)
