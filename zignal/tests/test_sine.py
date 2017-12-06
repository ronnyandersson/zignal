'''
Created on 26 Oct 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
'''

# standard library
from __future__ import division, print_function
import unittest

# external libraries
import nose

# local libraries
from zignal import Sinetone, SquareWave

class Test_Sinetone(unittest.TestCase):
    def test_endpoint(self):
        fs = 100
        f0 = 1
        x = Sinetone(f0=f0, fs=fs, duration=1, gaindb=20)
        print(x)
        print(x.samples[-5:,:])
        # We've created a one period sine. The last sample should not be
        # very close to zero. If it were, then a concatenation of two
        # sines would mean that we have one zero value too many at the
        # concatenation point --> discontinuity 
        self.assertNotAlmostEqual(x.samples[-1], 0, places=5)
        
    def test_center_frequency(self):
        fs = 48000
        f0 = 997
        x = Sinetone(f0=f0, fs=fs, duration=2, gaindb=20)
        print(x)
        freq, mag = x.fft(window="rectangular")
        
        self.assertAlmostEqual(freq[mag.argmax()], f0, places=7)
        
class Test_SetSampleRate(unittest.TestCase):
    def setUp(self):
        self.fs     = 1000
        self.dur    = 2.0
        self.f0     = 100
        self.x      = Sinetone(f0=self.f0, fs=self.fs, duration=self.dur, gaindb=-10)
        
        print(self.id())
        print('Before:\n%s' %self.x)
        
    def test_duration_fs_up_2_5(self):
        self.assertAlmostEqual(self.x.duration, self.dur, places=5)
        
        self.x.set_sample_rate(self.fs*2.5)
        print('After:\n%s' %self.x)
        
        self.assertAlmostEqual(self.x.duration, self.dur/2.5, places=5)
        
    def test_duration_fs_down_2_5(self):
        self.assertAlmostEqual(self.x.duration, self.dur, places=5)
        
        self.x.set_sample_rate(self.fs/2.5)
        print('After:\n%s' %self.x)
        
        self.assertAlmostEqual(self.x.duration, self.dur*2.5, places=5)
        
    def test_frequency_fs_down_3(self):
        self.assertAlmostEqual(self.x.f0, self.f0, places=5)
        
        self.x.set_sample_rate(self.fs/3)
        print('After:\n%s' %self.x)
        
        self.assertAlmostEqual(self.x.f0, self.f0/3, places=5)
        
    def test_frequency_fs_up_3(self):
        self.assertAlmostEqual(self.x.f0, self.f0, places=5)
        
        self.x.set_sample_rate(self.fs*3)
        print('After:\n%s' %self.x)
        
        self.assertAlmostEqual(self.x.f0, self.f0*3, places=5)
        
class Test_SetSampleRate_Square(Test_SetSampleRate):
    def setUp(self):
        self.fs     = 10000
        self.dur    = 3.0
        self.f0     = 500
        self.x      = SquareWave(f0=self.f0, fs=self.fs, duration=self.dur, gaindb=-10)
        
if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s "+
                                 "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
