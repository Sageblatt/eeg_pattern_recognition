import numpy as np
import conversion as c
import filtering as f
import candidates as ca
import massacre as m
import neuralnetwork as nn
import os
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
tk.Tk().withdraw()


if __name__ == '__main__':
    DELETE_PREV_FILES = True
    USE_GUI = True

    if DELETE_PREV_FILES:
        try:
            for s in range(10):
                os.remove(f'data/main{s}.csv')
        except FileNotFoundError:
            print('Files not found, nothing to remove')
    
    if USE_GUI:
        print('Explorer window opened')
        fn = askopenfilename()
    else:
        fn = 'data/raw/300.bdf'
        
    signals, freqs, channel_names = c.read_bdf(fn, show_output=True, extra_data=True, 
                                             show_annotations=False, downsample=True, ignore_accel=True)
    
    filtered = []
    dct = []
    spikes = []
    predict = []
    for s in range(len(signals)):
        sig_, _ = f.cut_low(signals[s], freqs[s])
        bool_filters = f.analyze(sig_, freqs[s])
        sig_, _ = f.denoise(sig_, freqs[s], bool_filters)
        sig_, _ = f.cut_low(sig_, freqs[s])
        filtered.append(sig_)
        dct = ca.candidates(filtered[s], freqs[s])
        cut = m.loading_data(filtered[s], freqs[s], dct)
        spikes.append(m.cutting_spikes(*cut))
        m.saving_data(f'data/main{s}.csv', spikes[s])
        predict.append(nn.model_predict(f'data/main{s}.csv'))
    
    print(predict[0])
    
    
    # plt.plot(filtered[0])