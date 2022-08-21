import numpy as np
import conversion as c
import filtering as f
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
tk.Tk().withdraw()


if __name__ == '__main__':
    fn = askopenfilename()
    signals, freqs, channel_names = c.read_bdf(fn, show_output=True, extra_data=True, 
                                             show_annotations=False, downsample=True, ignore_accel=True)
    
    filtered = []
    for s in range(len(signals)):
        sig_, _ = f.cut_low(signals[s], freqs[s])
        bool_filters = f.analyze(sig_, freqs[s])
        sig_, _ = f.denoise(sig_, freqs[s], bool_filters)
        sig_, _ = f.cut_low(sig_, freqs[s])
        filtered.append(sig_)
    
    plt.plot(filtered[0])