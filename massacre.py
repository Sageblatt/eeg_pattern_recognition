from EDFlib import edfreader
from math import floor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SPIKE_IN_THE_MIDDLE    = False
SPIKE_IN_THE_END       = False
SPIKE_IN_THE_BEGINNING = True

# TODO: - fix spike_in_the_middle
#       - fix spike_in_the_end



class cutter():
    
    def __init__(self):
        
        self.sig = np.array([])
        self.latency = np.array([])
        self.duration = np.array([])
        self.fs = 0
        self.t = np.array([])
        self.spikes = []
        self.starts = np.array([])
        self.ends = np.array([])

    def read_from_bdf(self, filename, channel):

        reader = edfreader.EdfReader(filename)
        ann = reader.read_annotations()
        sig = reader.get_signal(chn = channel)
        
        self.sig = np.array(sig)

        ann = [a for a in ann if a[2] == '']
        latency  = np.array(np.array(ann)[:, 0])
        duration = np.array(np.array(ann)[:, 1])
        
        self.latency  = np.array([float(i) for i in latency])
        self.duration = np.array([float(i) for i in duration])
        
        self.fs = reader.samplefrequency(channel)
        self.t = np.linspace(0, len(sig) / self.fs, len(sig))
        
        self.spikes = [self.latency, self.duration]


    def loading_data(self, sig, fs, data):

        """ 
            Loads information about spikes
            (while working with searching algo)
            
            Parameters:
                sig (np.array): signal
                fs (float)    : sampling frequency
                data (dict)   : latency and duration of spikes
        """
        self.sig = sig
        self.fs = fs
        self.t = np.linspace(0, len(sig) / fs, len(sig))

        data = pd.DataFrame(data)
        spikes = []
        spikes.append(data.loc[:, "latency"].to_numpy())
        spikes.append(data.loc[:, "duration"].to_numpy())
        self.spikes = np.array(spikes)


    def saving_data(self, fname, res):

        """
            Saves data to csv
            
            Parameters:
                fname (string) : name of file
                res (np.array) : data
        """

        pd.DataFrame(res).to_csv(fname, mode='a', index = False, 
                                                 header = False)

    def spike_in_the_beginning(self, sample_len, i):

        """
            Cuts sample with spike in the beginning
            
            Parameters:
                sample_len (int) : length of sample
                i (int)          : number of spike
                
            Return:
                start (int) : start of spike (time)
                end   (int) : end of spike (time)
        """
        spike_len = np.int32(np.floor(self.spikes[1][i]))
        start = np.searchsorted(self.t, self.spikes[0][i])

        end = start + spike_len + floor((sample_len - spike_len))
            
        if (end-start) != sample_len:
            end += sample_len - (end - start)
            
        return ([start, end])


    def spike_in_the_middle(self, sample_len, i):

        """
            Cuts sample with spike in the middle
            
            Parameters:
                sample_len (int) : length of sample
                i (int)          : number of spike
                
            Return:
                start (int) : start of spike (time)
                end   (int) : end of spike (time)
        """
            
        spike_len = floor(self.spikes[1][i])
        spike_start = (np.abs(self.t - self.spikes[0][i])).argmin()
            
        start =  spike_start - floor((sample_len - spike_len) / 2)
        end = spike_start + spike_len + floor((51 - spike_len) / 2)
            
        if (end-start) != sample_len:
            end += sample_len - (end - start)
            
        return([start, end])


    def spike_in_the_end(self, sample_len, i):

        """
            Cuts sample with spike in the end
            
            Parameters:
                sample_len (int) : length of sample
                i (int)          : number of spike
                
            Return:
                start (int) : start of spike (time)
                end   (int) : end of spike (time)
        """
        
        spike_len = floor(self.spikes[1][i])
        spike_start = (np.abs(self.t - self.spikes[0][i])).argmin()
        
        start = spike_start - floor((51 - spike_len))
        end = start + 51
        
        if (end-start) != sample_len:
            start -= sample_len - (end - start)

        return([start, end])

    @staticmethod
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
        spike = a * spike + b

        return spike


    def plot(data, time):

        """
            Draws sample of signal
            
            Parameters:
                data (np.array) : y
                time (np.array) : x
        """

        plt.plot(time, data, figure=plt.figure(figsize=(10.0, 6.0)))
        plt.show()


    def cutting_spikes(self, fname):

        """
            Cuts signal to samples with spikes
            
            Parameters:
                fname (str) : name of file to save
        """

        sample_len = 51

        res = []
        not_spikes = []
        starts = []
        ends = []

        for i in range(len(self.spikes[0])):
        
            spike = []
            time = []
            
            if (SPIKE_IN_THE_MIDDLE):
                data = self.spike_in_the_middle(sample_len, i)
                start, end = data[0], data[1]
            if (SPIKE_IN_THE_END):
                data = self.spike_in_the_end(sample_len, i)
                start, end = data[0], data[1]
            if (SPIKE_IN_THE_BEGINNING):   
                data = self.spike_in_the_beginning(sample_len, i) 
                start, end = data[0], data[1]
            if (end >= len(self.sig)):
                continue        
        
            starts.append(start)
            ends.append(end) 
               
            for n in range(start, end):
                spike.append(self.sig[n])
                time.append(self.t[n])
            
            #self.plot(spike, time)
            spike = np.array(spike)
            spike = self.normalization(np.array(spike))
            
            
            #self.plot(spike, time)
            
            res.append(spike.to_list())

        res = np.array(res, dtype=object)
        res = np.asanyarray(res)

        self.saving_data(fname, res)

        ends = [0] + ends

        self.starts = np.array(starts)
        self.ends = np.array(ends)        



    def cutting_not_spikes(self, fname):

        """
            Cuts signal to samples without spikes
            
            Parameters:
                fname (str) : name of file to save
        """

        not_spikes = []
        for i in range(len(self.starts)-1):
           if (self.starts[i] - self.ends[i]) >= 50:
                p = 0
                while ((self.starts[i] - (self.ends[i] + 50 * p)) >= 50):
                    not_spike = []
                    time = []
                    for k in range(self.ends[i] + 50 * p, 
                                   self.ends[i] + 50 * (p+1) + 1):
                        not_spike.append(self.sig[k])
                        time.append(self.t[k])
                    
                    #self.plot(not_spike, time)
                    not_spike = np.array(not_spike)   
                    not_spike = self.normalization(not_spike)
                    
                    not_spikes.append(not_spike) 
                    p += 1
                    
        not_spikes = np.array(not_spikes, dtype=object)
        not_spikes = np.asanyarray(not_spikes)
        
        self.saving_data(fname, not_spikes)

    def standalone(self, fname):

        for ch in range(1):
    
            print("number ", ch)
            self.read_from_bdf(fname, ch)
            
            fn  = 'data/spikes/channel_' + str(ch+1) + '.csv'
            fn1 = 'data/not_spikes/channel_' + str(ch+1) + '.csv'
            
            self.cutting_spikes(fn)
            self.cutting_not_spikes(fn1)



if __name__ == '__main__':

    fname = input('Enter .bdf file name: ')
    #fname = '24h_spikes.bdf'
    c = cutter()
    c.standalone(fname)

