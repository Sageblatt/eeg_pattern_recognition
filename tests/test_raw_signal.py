import os.path, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import unittest
from raw_signal import Signal

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import matplotlib.pyplot as plt


class TestRawSignal(unittest.TestCase):
    def setUp(self):
        self.s = np.array([1., 2., 3.])
        self.fs = 5
    
    def test_constructor(self):
        a = Signal(self.s, self.fs)
        self.assertIsInstance(a, Signal)
    
    def test_nd_arrays(self):
        d = np.array([[1., 2.], [3., 4.]])
        with self.assertRaises(TypeError):
            Signal(d, self.fs)
    
    def test_slicing(self):
        a = Signal(self.s, self.fs)
        self.assertEqual(self.fs, a.fs)
        self.assertIsNone(assert_array_equal(a[1:], self.s[1:]))

    def test_view_cast(self):
        with self.assertRaises(TypeError):
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

        self.assertIsNone(assert_almost_equal(s, y1, decimal=2))
        
            

if __name__ == '__main__':
    unittest.main()