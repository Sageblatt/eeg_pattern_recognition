import massacre as m

if __name__ == '__main__':

    fname = '24h_spikes.bdf'
    for ch in range(6):
    
        print("number ", ch)
        data = m.read_from_bdf(fname, ch)
        sig = data[0]
        spikes = data[1]
        t = data[2]
        
        fn  = 'data/spikes/channel_' + str(ch+1) + '.csv'
        fn1 = 'data/not_spikes/channel_' + str(ch+1) + '.csv'
        
        starts, ends = m.cutting_spikes(sig, spikes, t, fn)
        m.cutting_not_spikes(starts, ends, sig, t, fn1)
