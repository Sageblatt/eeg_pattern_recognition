from edflib import edfreader
from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SPIKE_IN_THE_MIDDLE    = False
SPIKE_IN_THE_END       = False
SPIKE_IN_THE_BEGINNING = True

def read_from_bdf(filename, channel):

    reader = edfreader.EdfReader(filename)
    ann = reader.read_annotations()
    sig = reader.get_signal(chn = channel)
    
    sig      = np.array(sig)

    ann = [a for a in ann if a[2] == '']
    latency  = np.array(np.array(ann)[:, 0])
    duration = np.array(np.array(ann)[:, 1])
    
    latency  = np.array([float(i) for i in latency])
    duration = np.array([float(i) for i in duration])
    
    fs = reader.samplefrequency(channel)
    t = np.linspace(0, len(sig) / fs, len(sig))
    
    spikes = [latency, duration]
    
    return ([sig, spikes, t])

def loading_data(sig, fs, data):

    """ 
        Loads information about spikes
        
        Parameters:
            sig (np.array): signal
            fs (float)    : sampling frequency
            data (dict)   : latency and duration of spikes
        Return:
            sig (np.array)   : signal
            spikes (np.array): latency and duration of spikes
            t (np.array)     : time array
    """
    
    #sig_ = np.load('data/np_filt/IIS Channel_' + str(sig_num) + '_filt.npy')
    #fs = sig_[0]
    #sig = sig_[1:]
    
    t = np.linspace(0, len(sig) / fs, len(sig))

    #data = pd.read_excel('data/Spikes.xlsx', index_col = 0)
    
    data = pd.DataFrame(data)
    spikes = []
    spikes.append(data.loc[:, "latency"].to_numpy())
    spikes.append(data.loc[:, "duration"].to_numpy())
    spikes = np.array(spikes)
    
    return([sig, spikes, t])


def saving_data(fname, res):

    """
        Saves data to csv
        
        Parameters:
            fname (string) : name of file
            res (np.array) : data
    """

    pd.DataFrame(res).to_csv(fname, mode='a', index = False, header = False)

def spike_in_the_middle(spikes, sample_len, i, t):

    """
        Cuts sample with spike in the middle
        
        Parameters:
            spikes (np.array): latency and duration of spikes 
            sample_len (int) : length of sample
            i (int)          : number of spike
            t (np.array)     : time array
            
        Return:
            start (int) : start of spike (time)
            end   (int) : end of spike (time)
    """
        
    spike_len = floor(spikes[1][i])
    spike_start = (np.abs(t - spikes[0][i])).argmin()
        
    start =  spike_start - floor((sample_len - spike_len) / 2)
    end = spike_start + spike_len + floor((51 - spike_len) / 2)
        
    if (end-start) != sample_len:
        end += sample_len - (end - start)
        
    return([start, end])

    
def spike_in_the_end(spikes, sample_len, i, t):

    """
        Cuts sample with spike in the end
        
        Parameters:
            spikes (np.array): latency and duration of spikes 
            sample_len (int) : length of sample
            i (int)          : number of spike
            t (np.array)     : time array
            
        Return:
            start (int) : start of spike (time)
            end   (int) : end of spike (time)
    """
    
    spike_len = floor(spikes[1][i])
    spike_start = (np.abs(t - spikes[0][i])).argmin()
    
    start = spike_start - floor((51 - spike_len))
    end = start + 51
    
    if (end-start) != sample_len:
        start -= sample_len - (end - start)

    return([start, end])
 
    
def spike_in_the_beginning(spikes, sample_len, i, t):

    """
        Cuts sample with spike in the beginning
        
        Parameters:
            spikes (np.array): latency and duration of spikes 
            sample_len (int) : length of sample
            i (int)          : number of spike
            t (np.array)     : time array
            
        Return:
            start (int) : start of spike (time)
            end   (int) : end of spike (time)
    """

    spike_len = floor(spikes[1][i])
    start = (np.abs(t - spikes[0][i])).argmin()
    end = start + spike_len + floor((51 - spike_len))
        
    if (end-start) != sample_len:
        end += sample_len - (end - start)
        
    return([start, end])
    

def normalization(spike):

    """
        Makes signal's value from 0 to 1
        
        Patameters:
            spike (list) : signal
        
        Return:
            spike (list) : normalized signal
    """
    
    up = np.max(spike)
    down = np.min(spike)
    b = (up + down) / (up - down)
    a = (1 - b) / down
    spike = a * np.array(spike) + b
    return(spike)


def plot(data, time):

    """
        Draws sample of signal
        
        Parameters:
            data (np.array) : y
            time (np.array) : x
    """

    plt.plot(time, data, figure=plt.figure(figsize=(10.0, 6.0)))
    plt.show()


def cutting_spikes(sig, spikes, t, fname):

    """
        Cuts signal to samples with spikes
        
        Parameters:
            sig (np.array) : signal
            spikes (list)  : latency and duration of spikes
            t (np.array)   : time
    """

    sample_len = 51

    res = []
    not_spikes = []
    starts = []
    ends = []

    for i in range(len(spikes[0])):
    
        spike = []
        time = []
        
        if (SPIKE_IN_THE_MIDDLE):
            data = spike_in_the_middle(spikes, sample_len, i, t)
            start, end = data[0], data[1]
        if (SPIKE_IN_THE_END):
            data = spike_in_the_end(spikes, sample_len, i, t)
            start, end = data[0], data[1]
        if (SPIKE_IN_THE_BEGINNING):   
            data = spike_in_the_beginning(spikes, sample_len, i, t) 
            start, end = data[0], data[1]
        if (end >= len(sig)):
            continue        
    
        starts.append(start)
        ends.append(end) 
           
        for n in range(start, end):
            spike.append(sig[n])
            time.append(t[n])
        
        #plot(spike, time)

        spike = normalization(spike)
        
        
        #plot(spike, time)
        
        res.append(spike)

    res = np.array(res, dtype=object)
    res = np.asanyarray(res)

    saving_data(fname, res)

    ends = [0] + ends
    
    return(starts, ends)


def cutting_not_spikes(starts, ends, sig, t, fname):

    """
        Cuts signal to samples without spikes
        
        Parameters:
            starts (list) : starts of spikes (time)
            ends (list)   : ends of spikes (time)
            sig (np.array): signal
            t (np.array)  : time
            fname (str)   : name of file to save
    """

    not_spikes = []
    for i in range(len(starts)-1):
       if (starts[i] - ends[i]) >= 50:
            p = 0
            while ((starts[i] - (ends[i] + 50 * p)) >= 50):
                not_spike = []
                time = []
                for k in range(ends[i] + 50 * p, ends[i] + 50 * (p+1) + 1):
                    not_spike.append(sig[k])
                    time.append(t[k])
                
                #plot(not_spike, time)
                   
                not_spike = normalization(not_spike)
                
                not_spikes.append(not_spike) 
                p += 1
                
    not_spikes = np.array(not_spikes, dtype=object)
    not_spikes = np.asanyarray(not_spikes)
    
    saving_data(fname, not_spikes)

if __name__ == '__main__':
    read_from_bdf("24h_spikes.bdf", 3)

# TODO: rework main to use this script standalone as it was in earlier versions
# if __name__ == '__main__':   
#     # Deletes previously made files 
#     # to protect data from repeating
    
#     DELETE_PREV_FILES = True

#     if DELETE_PREV_FILES:
#         try:
#             os.remove('data/not_spikes.csv')
#             os.remove('data/spikes.csv')
#         except FileNotFoundError:
#             print('Files not found, nothing to remove')
            
    
#     for sig_num in range(1, 7):
#         if sig_num == 5:
#             continue
        
#         # in final main file signal will be loaded before 
#         # using 'candidates'
    
#         data   = loading_data(sig_num, candidates(sig), sig)
#         sig    = data[0]
#         spikes = data[1]
#         t      = data[2]
        
#         data   = cutting_spikes(sig, spikes, t)
#         starts = data[0] 
#         ends   = data[1]
        
#         cutting_not_spikes(starts, ends, sig, t)

