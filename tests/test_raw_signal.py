import inc_path
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from raw_signal import Signal


class TestRawSignal(unittest.TestCase): # TODO: write messages msg
    def setUp(self):
        self.s = np.array([1., 2., 3.])
        self.fs = 5
    
    def test_constructor(self):
        a = Signal(self.s, self.fs)
        self.assertIsInstance(a, Signal, 'Constructor did not succeed.')
    
    def test_nd_arrays(self):
        d = np.array([[1., 2.], [3., 4.]])
        with self.assertRaises(TypeError, msg='Expected TypeError when creating'
                               'Signal from array with more than 1 dimensions.'):
            Signal(d, self.fs)
    
    def test_slicing(self):
        a = Signal(self.s, self.fs)
        self.assertEqual(self.fs, a.fs, 'Sampling frequency was corrupted during '
                         'constructon of the Signal object.')
        self.assertIsNone(assert_array_equal(a[1:], self.s[1:]), 'Slices of the '
                          'source array and created Signal object do not match.')

    def test_view_cast(self):
        with self.assertRaises(TypeError, msg='Expected TypeError when trying '
                               'to view-cast ndarray to Signal type.'):
            self.s.view(Signal)
            
    def test_resample(self):
        N = 10_000
        left = 0
        right = 10
        modifier = 0.95
        y = np.sin(np.linspace(left, right, N))
        
        s = Signal(y, (right - left)/(N - 1))
        s.resample(s.fs*modifier)
        y1 = np.sin(np.linspace(left, right, s.size))

        self.assertIsNone(assert_almost_equal(s, y1, decimal=2), 'Expected signals '
                          'to be closer than 1e-2 after resampling.')
        
            

if __name__ == '__main__':
    unittest.main()