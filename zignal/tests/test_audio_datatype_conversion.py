'''
Created on 28 Feb 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
'''

# standard library
from __future__ import division, print_function
import unittest

# external libraries
import numpy as np
import nose

# local libraries
from zignal import Audio
        
class Test_ConvertBackToBack(unittest.TestCase):
    def setUp(self):
        self.x = Audio(fs=10, initialdata=np.zeros((10, 1)))
        self.x.samples[0] =  -1.0
        self.x.samples[1] =   1.0
        print(self.x)
        print(self.x.samples)
        print()
        
    def quantization_step_size(self, bits):
        return 2**-(bits-1)
    
    def test_float_to_int8_to_float64(self):
        self.x.convert_to_integer(targetbits=8)
        self.x.convert_to_float(targetbits=64)
        print(self.x)
        print(self.x.samples)
        
        q = self.quantization_step_size(8)
        self.assertAlmostEqual(self.x.samples[0], -1.0 + q, places=20)
        self.assertAlmostEqual(self.x.samples[1],  1.0 - q, places=20)
        
    def test_float_to_int16_to_float64(self):
        self.x.convert_to_integer(targetbits=16)
        self.x.convert_to_float(targetbits=64)
        print(self.x)
        print(self.x.samples)
        
        q = self.quantization_step_size(16)
        self.assertAlmostEqual(self.x.samples[0], -1.0 + q, places=20)
        self.assertAlmostEqual(self.x.samples[1],  1.0 - q, places=20)
        
    def test_float_to_int32_to_float64(self):
        self.x.convert_to_integer(targetbits=32)
        self.x.convert_to_float(targetbits=64)
        print(self.x)
        print(self.x.samples)
        
        q = self.quantization_step_size(32)
        self.assertAlmostEqual(self.x.samples[0], -1.0 + q, places=20)
        self.assertAlmostEqual(self.x.samples[1],  1.0 - q, places=20)
        
class Test_ConvertFloatToInt(unittest.TestCase):
    def setUp(self):
        self.x = Audio(fs=10, initialdata=np.zeros((10, 1)))
        self.x.samples[0] =  -1.0
        self.x.samples[1] =   1.0
        print(self.x)
        print(self.x.samples)
        print()
        
    def convert(self, targetbits=None):
        self.x.convert_to_integer(targetbits=targetbits)
        print(self.x)
        print(self.x.samples)
        
        self.assertIsInstance(self.x.samples, np.ndarray)
        
    def test_int8(self):
        self.convert(targetbits=8)
        self.assertTrue(self.x.samples.dtype==np.int8)
        
        # 8 bits 2's complement
        # min is -128
        # max is  127
        self.assertEquals(self.x.samples[0], -127) # must be symmetrical
        self.assertEquals(self.x.samples[1],  127) # must be symmetrical
        
    def test_int16(self):
        self.convert(targetbits=16)
        self.assertTrue(self.x.samples.dtype==np.int16)
        
        # 16 bits 2's complement
        # min is -32768
        # max is  32767
        self.assertEquals(self.x.samples[0], -32767) # must be symmetrical
        self.assertEquals(self.x.samples[1],  32767) # must be symmetrical
        
    def test_int32(self):
        self.convert(targetbits=32)
        self.assertTrue(self.x.samples.dtype==np.int32)
        
        # 32 bits 2's complement
        # min is -2147483648
        # max is  2147483647
        self.assertEquals(self.x.samples[0], -2147483647) # must be symmetrical
        self.assertEquals(self.x.samples[1],  2147483647) # must be symmetrical
        
class Test_ConvertIntToFloat32(unittest.TestCase):
    def convert(self, y, targetbits=None):
        print(y)
        print(y.pretty_string_samples())
        print()
        
        y.convert_to_float(targetbits=targetbits)
        print(y)
        print(y.pretty_string_samples())
        print()
        
    def test_int8(self):
        x = Audio(fs=10, initialdata=np.zeros((10, 1), dtype=np.int8))
        x.samples[0] = -128
        x.samples[1] =  127
        
        self.convert(x, targetbits=32)
        
        self.assertAlmostEqual(x.samples[0], -1.0,      places=20)
        self.assertAlmostEqual(x.samples[1], 127/128,   places=20)
        
    def test_int16(self):
        x = Audio(fs=10, initialdata=np.zeros((10, 1), dtype=np.int16))
        x.samples[0] = -32768
        x.samples[1] =  32767
        
        self.convert(x, targetbits=32)
        
        self.assertAlmostEqual(x.samples[0], -1.0,          places=20)
        self.assertAlmostEqual(x.samples[1], 32767/32768,   places=20)
        
    def test_int32(self):
        x = Audio(fs=10, initialdata=np.zeros((10, 1), dtype=np.int32))
        x.samples[0] = -2147483648
        x.samples[1] =  2147483647
        
        self.convert(x, targetbits=32)
        
        # resolution is lost in this conversion, however we should be close enough.
        self.assertAlmostEqual(x.samples[0], -1.0, places=20) # exact value
        self.assertAlmostEqual(x.samples[1],  1.0, places=20) # close enough
        
if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s "+
                                 "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
