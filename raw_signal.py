import numpy as np
import inspect

from scipy.signal import resample as sp_resample



class Signal(np.ndarray):
    """
        Represents signal as numpy array with new attribute — sampling 
        frequency (Signal.fs) and several new methods.
    """
    def __new__(cls, input_array: np.ndarray, fs: float | int):
        if input_array.ndim > 1:
            raise TypeError('Signal supports only 1D arrays.')
        
        obj = np.asarray(input_array).view(cls).copy() # TODO: mention COPIES!
    
        obj.fs = fs
        
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return

        self.fs = getattr(obj, 'fs', None)
        
        if self.fs is None and inspect.stack()[1][3] != '__new__':
            raise TypeError('numpy.ndarray cannot be view-casted to Signal. '
                            'E.g. "a = numpy.array([1, 2]).view(Signal)" will '
                            'raise an exception. Consider creating new Signal '
                            'object from existing array.')
        return
    
    def __str__(self):
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
    
    def pts(self, left_idx: int = 0, right_idx: int = -1):
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
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    
    s = Signal(y, 3000)
    
    plt.plot(s.get_time(), s, 'b-')
    s.resample(1500)
    plt.plot(*s.pts(2, 3), "kv")
    print(s.fs)
    
    
    
    