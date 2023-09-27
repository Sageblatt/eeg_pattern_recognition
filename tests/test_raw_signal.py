import inc_path
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from raw_signal import Signal


class TestRawSignal(unittest.TestCase):
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
    
    def test_normalizing(self):
        x = np.linspace(0, 10, 500)
        
        p = np.sin(x)
        r = 5 * np.sin(x)
        
        s1 = Signal(p, 100)
        s2 = Signal(r, 100)
        
        s2.normalize()
        self.assertIsNone(assert_almost_equal(s2, s1, decimal=2), 'Expected signals '
                          'to be closer than 1e-2 after normalization.')
        
        
    def test_high_pass(self):
        def plateau(x):
            if x < 1:
                return 0
            elif x > 8:
                return 0
            else:
                return 1
            
        p_vec = np.vectorize(plateau)
        
        a = np.linspace(0, 10, 500)
        b = p_vec(a)
        
        c = Signal(b, 20)
        c.apply_hp(1, False)
        
        # Without multiprocessing
        xf, yf = c.get_spectrum()
        zeros = np.zeros(xf[xf < 1].size)
        self.assertIsNone(assert_almost_equal(yf[xf < 1], zeros, decimal=3),
                          'Expected signals frequency components below 1 Hz to '
                          ' to zero WITHOUT multiprocessing.')
        
        # With multiprocessing
        c = Signal(b, 20)
        c.apply_hp(1, True)
        
        xf, yf = c.get_spectrum()
        zeros = np.zeros(xf[xf < 1].size)
        self.assertIsNone(assert_almost_equal(yf[xf < 1], zeros, decimal=3),
                          'Expected signals frequency components below 1 Hz to '
                          ' to zero WITH multiprocessing.')
    
    # def test_denoising(self): # May be too complex to test
    #     def plateau(x):
    #         if x < 0.6:
    #             return 0
    #         elif x > 0.8:
    #             return 0
    #         else:
    #             return 1
            
    #     p_vec = np.vectorize(plateau)
        
    #     a = np.linspace(0, 1, 250)
    #     b = p_vec(a) + 2*np.sin(2*np.pi*25*a) + 7*np.sin(2*np.pi*50*a) + \
    #         4*np.sin(2*np.pi*100*a)
            
    #     c = Signal(b, 250) 
    #     c.denoise()
        
        

if __name__ == '__main__':
    unittest.main()