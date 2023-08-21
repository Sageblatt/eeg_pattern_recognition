import inc_path
import unittest

from filecmp import cmp
import os.path
import os
from EDFlib.edfreader import EDFexception

from record import read_file


class TestRecord(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_read(self):
        r = read_file('test.bdf', get_annots=False)
        self.assertEqual(r.size, 2, f'Read {r.size} signals, expected 2.')
        self.assertEqual(r.signals[0].size, r.signals[1].size, 
                         'Length of signals does not match.')
        self.assertEqual(r.signals[0].size, 10250, 
                         f'Read {r.signals[0].size} samples, expected 10250.')
        self.assertEqual(r.signals[0].fs, r.signals[1].fs, 
                         'Sampling frequencies of signals do not match.')
        self.assertEqual(r.signals[0].fs, 250, 
                         f'Sampling frequency of Channel 1 is {r.signals[0].fs} Hz, '
                         'expected 250 Hz.')
        self.assertEqual(r.annotations, None, 'Expected 0 annotatins to be read.')
    
    def test_annots(self):
        r = read_file('test.bdf', get_annots=True)
        self.assertEqual(len(r.annotations), 2, 'Expected 2 annotations in the'
                         f'file, got {len(r.annotations)}.')
        self.assertEqual(r.annotations[1].description, 'B1',
                         'Expected annotation description is "B1", not '
                         f'{r.annotations[1].description}')
        
    def test_invalid_read(self):
        with self.assertRaises(EDFexception, 
                               msg='Expected EDFexception to be risen when'
                               'reading non-existing file.'):
            read_file('non-existing_file.bdf')
        
    def test_write(self):
        r = read_file('test.bdf', get_annots=True)
        r.write_file('output.bdf', current_time=False)
        self.assertTrue(cmp('test.bdf', 'output.bdf', shallow=False), 
                        'Expected equal files after consecutive read and write.')
    
    def tearDown(self):
        if os.path.exists('output.bdf'):
            os.remove('output.bdf')
        
            

if __name__ == '__main__':
    unittest.main()