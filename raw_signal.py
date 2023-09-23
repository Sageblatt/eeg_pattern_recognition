import numpy as np
import inspect
import scipy.fft as fft

from multiprocessing import Pool, cpu_count
import itertools

from scipy.signal import buttord, butter, sosfilt, find_peaks
from scipy.signal import resample as sp_resample


class Signal(np.ndarray):
    """
        Represents signal as numpy array with new attribute — sampling 
        frequency (Signal.fs) and several new methods. Copies the array with
        numpy.ndarray.copy().
    """
    def __new__(cls, input_array: np.ndarray, fs: float | int):
        """
        Method to correctly create np.ndarray subclass.

        Parameters
        ----------
        input_array : np.ndarray
            1-D array containing signal data.
        fs : float | int
            Sampling frequency.

        Raises
        ------
        TypeError
            Only 1D arrays are supported.

        Returns
        -------
        obj : Signal
            Created object.

        """
        if input_array.ndim > 1:
            raise TypeError('Signal supports only 1D arrays.')
        
        obj = np.asarray(input_array).view(cls).copy()
    
        obj.fs = fs
        
        return obj

    def __array_finalize__(self, obj):
        """
        Extra method for correct numpy.ndarray subclassing.
        """
        if obj is None: 
            return

        self.fs = getattr(obj, 'fs', None)
        
        if self.fs is None and inspect.stack()[1][3] != '__new__':
            raise TypeError('numpy.ndarray cannot be view-casted to Signal. '
                            'E.g. "a = numpy.array([1, 2]).view(Signal)" will '
                            'raise an exception. Consider creating new Signal '
                            'object from existing array.')
        return
    
    def __reduce__(self):
        """ 
        Extra method for pickling, which is used in multiprocessing.
        """
        pickled_state = super().__reduce__()
        new_state = pickled_state[2] + (self.__dict__,)
        return (pickled_state[0], pickled_state[1], new_state)
    
    def __setstate__(self, state):
        """ 
        Extra method for pickling, which is used in multiprocessing.
        """
        self.__dict__.update(state[-1])
        super().__setstate__(state[0:-1])
    
    def __init__(self, input_array: np.ndarray, fs: float | int):
        """
        See __new__ method.

        Parameters
        ----------
        input_array : np.ndarray
            1-D array containing signal data.
        fs : float | int
            Sampling frequency.

        """
        pass
    
    def __str__(self):
        """
        Allows to print Signal object in the same manner as numpy.ndarray.
        """
        s1 = super().__str__()
        return s1 + f'\nSampling frequency:  {self.fs} Hz'
    
    def get_time(self) -> np.ndarray:
        """
        Get an array of timestamps for this signal.

        Returns
        -------
        time_array: np.ndarray
            Array of timestamps.

        """
        return np.linspace(0, self.size/self.fs, self.size)
    
    def resample(self, new_freq: float | int) -> None:
        """
        Change the sampling frequency of a signal. Uses scipy.signal.resample.

        Parameters
        ----------
        new_freq : float | int
            Desired sampling frequency.

        Returns
        -------
        None

        """
        time = (self.size-1) / self.fs
        new_sig = Signal(sp_resample(self, int(time*new_freq)), 
                         int(time*new_freq)/time)
        self.resize(new_sig.size, refcheck=False)
        self[:] = new_sig
        self.fs = new_sig.fs
        return
    
    def pts(self, left_idx: int = 0,
            right_idx: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """
        Get time array and signal data as tuple to use in 
        matplotlib.pyplot.plot(). Slicing is also available via optional 
        arguments.

        Parameters
        ----------
        left_idx : int, optional
            Left index of slice. The default is 0.
        right_idx : int, optional
            Right index of slice. The default is -1.

        Returns
        -------
        A tuple with the following attributes:
            
        time_array: np.ndarray
            See Signal.get_time().
        signal: np.ndarray
            Signal data array.
            
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        
        >>> arr = np.array([1., 2., 3.])
        >>> s = Signal(arr, 5)
        
        >>> plt.plot(*s.pts())

        """
        return self.get_time()[left_idx:right_idx], self[left_idx:right_idx]
    
    
    @staticmethod
    def _high_pass(signal: 'Signal', threshold: int | float) -> 'Signal':
        """
        Function that cuts frequencies lower than threshold 
        for given signal slice.

        Parameters
        ----------
        signal : 'Signal'
            Signal to be processed.
        threshold : int | float
            Frequency threshold.

        Returns
        -------
        'Signal'
            Signal with applied filter.

        """
        x_f = fft.rfftfreq(signal.size, 1 / signal.fs)
        y_f = fft.rfft(signal)
        y_f[(x_f <= threshold)] = 0
        return Signal(fft.irfft(y_f), signal.fs)
    
    
    def apply_hp(self, threshold: int | float, parallel: bool = False) -> None: # TODO: windows
        """
        Removes high pass filter to the whole signal.

        Parameters
        ----------
        threshold : int | float
            Threshold of the filter.
        parallel : bool, optional
            If `True` will use all CPU cores to compute the result.
            The default is `False`.

        Returns
        -------
        None

        """
        window_size = 10
        delta = self.size % window_size
        
        cut_sig = np.array_split(self[:-delta-window_size],
                                 self.size // window_size - 1)
        
        cut_sig.append(self[-delta-window_size:])
        
        
        if parallel:
            pool = Pool(cpu_count())
            cut_sig = pool.starmap(Signal._high_pass,
                                   zip(cut_sig, itertools.repeat(threshold)))
            pool.close()
            pool.join()
        else:
            for i in range(len(cut_sig)):
                cut_sig[i] = Signal._high_pass(cut_sig[i], threshold)
                
        self[:] = np.concatenate(cut_sig)
        return
    
    def denoise(self) -> None:
        """
        Removes electical hum (50 Hz and divisible by 25 Hz frequencies) from
        whole signal.

        Raises
        ------
        NotImplementedError
            When signal's sampling frequency is higher than 250 Hz filter
            for 125Hz peak may work incorrectly.

        Returns
        -------
        None

        """
        xf = fft.rfftfreq(self.size, 1 / self.fs) # TODO: adaptive algorithm using iirnotch
        yf = np.abs(fft.rfft(self))
        
        freqs = [25, 50, 75, 100, 125]
        coefs = [5, 3, 5, 3, 5]
        bool_freqs = np.zeros(len(freqs))
        
        for i in range(len(freqs)):
            if np.max(yf[np.abs(xf - freqs[i]) < 0.3]) > \
                coefs[i] * np.mean(yf[(0.5 < np.abs(xf - freqs[i])) & (np.abs(xf - freqs[i]) < 1.5)]):
                bool_freqs[i] = 1
        
        sos = []
        
        if bool_freqs[0]:
            ord, wn = buttord([24.5, 26.5], [24.9, 25.1], 2, 5, fs=self.fs)
            sos.append(butter(ord, wn, 'bs', fs=self.fs, output='sos'))
        if bool_freqs[1]:
            ord, wn = buttord([49, 51], [49.9, 50.1], 5, 30, fs=self.fs)
            sos.append(butter(ord, wn, 'bs', fs=self.fs, output='sos'))
        if bool_freqs[2]:
            ord, wn = buttord([74.5, 76.5], [74.9, 75.1], 2, 5, fs=self.fs)
            sos.append(butter(ord, wn, 'bs', fs=self.fs, output='sos'))
        if bool_freqs[3]:
            ord, wn = buttord([99, 101], [99.95, 100.05], 3, 30, fs=self.fs)
            sos.append(butter(ord, wn, 'bs', fs=self.fs, output='sos'))
        if bool_freqs[4]:
            if self.fs > 252:
                raise NotImplementedError('Sampling frequency is higher than '
                                          '250 Hz. Unsupported.')
            ord, wn = buttord(124.7, 124.8, 15, 25, fs=self.fs)
            sos.append(butter(ord, wn, 'lp', fs=self.fs, output='sos'))
        
        for f in sos:
            self[:] = sosfilt(f, self[:])
    
        return
    
    def get_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get two arrays, containing signals spectrum: frequency '
        'array and amplitudes.

        Returns
        -------
        np.ndarray, numpy.ndarray
            First array -- frequencies, second one -- amplitudes.

        """
        return fft.rfftfreq(self.size, 1 / self.fs), np.abs(fft.rfft(self))
    
    
    def get_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds signal's segments that are most likely to be spikes.

        Returns
        -------
        duration : numpy.ndarray
            Durations of alleged spikes (number of samples in each
            alleged spike).
        latency : numpy.ndarray
            Starting points for each alleged spike (in seconds, the record 
            starts at 0 seconds).

        """
        t = self.get_time()
        
        high_peaks_indexes = find_peaks(self)[0]
        high_peaks = self[high_peaks_indexes]
        
        low_peaks_indexes = find_peaks(self*-1)[0]
        low_peaks = self[low_peaks_indexes]
        
        lower_size = min(high_peaks.size, low_peaks.size)
        delta = high_peaks[:lower_size] - low_peaks[:lower_size]
        delta = delta[delta > 0.05]
                    
        width_avg = np.mean(delta)
        spikes_num = 0
        spikes = []
        times = []
        i = 0
        while i < len(high_peaks) - 2:
            if abs(high_peaks[i] - low_peaks[i]) >= 4 * width_avg or abs(high_peaks[i] - low_peaks[i+1]) >= 4 * width_avg:
                spikes_num = spikes_num + 1
                spike = []
                time = None
                
                if i >= 2 and i + 5 <= len(high_peaks) - 1:
                    for j in self[high_peaks_indexes[i-2]:high_peaks_indexes[i+5]]:
                        spike.append(j)
                    time = t[high_peaks_indexes[i-2]]
                elif i < 2 or i + 5 > len(high_peaks) - 1:
                    if i < 2:
                        delt1 = 0
                    else:
                        delt1 = i - 2
                    if i + 5 > len(high_peaks) - 1:
                        delt2 = len(high_peaks) - i - 1
                    else:
                        delt2 = 4
                    for j in self[high_peaks_indexes[delt1]:high_peaks_indexes[i+delt2]]:
                        spike.append(j)
                    time = t[high_peaks_indexes[delt1]]
                spikes.append(spike)
                times.append(time)
                i = i + 4
            else:
                i = i + 1
        
        duration = np.array([len(spikes[i]) for i in range(len(spikes))])
        latency = np.array(times)

        return duration, latency

    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fs = 250
    x = np.arange(0, 5, 1/fs)
    # y = np.random.rand(x.size) + np.sin(2*np.pi*50*x)*15
    y = 25 * x* np.sin(2*np.pi*1*x) * \
        np.sin(2*np.pi*5*x) * np.cos(2*np.pi*7*np.cos(2*np.pi*3.4*x))
    
    s = Signal(y, fs)
    
    plt.plot(*s.pts(), "kv")
    s.apply_hp(1, True)
    # s.denoise()
    plt.plot(*s.pts(), "ro")
    
    fig = plt.figure(figsize=(9., 6.), dpi=300)
    
    plt.plot(*s.get_spectrum())
    print(s.get_candidates()) # out is 15, 25, 16, 23 probably this code
    # snippet should be included in tests

    
    
    
    
