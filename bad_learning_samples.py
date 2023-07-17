import massacre as m

if __name__ == '__main__':
    fname = '24h_spikes.bdf'
    for ch in range(6):
        data = m.read_from_bdf(fname, ch)
        sig = data[0]
        spikes = data[1]
        t = data[2]
        
        file = 'data/channel_' + str(ch+1) + '.csv'
        m.cutting_spikes(sig, spikes, t, file)
