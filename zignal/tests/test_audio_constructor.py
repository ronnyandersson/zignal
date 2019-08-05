'''
Created on 28 Feb 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
'''

# standard library
import unittest

# external libraries
import numpy as np
import nose

# local libraries
from zignal import Audio

class Test_EmptyConstructor(unittest.TestCase):
    def setUp(self):
        self.x = Audio()
        print(self.x)

    def test_default_constructor(self):
        self.assertAlmostEqual(self.x.fs, 96000, places=7)
        self.assertEqual(self.x.ch, 0)
        self.assertEqual(self.x.nofsamples, 0)
        self.assertEqual(self.x.duration, 0)
        self.assertIsInstance(self.x.samples, np.ndarray)

    def test_str_method(self):
        self.assertIsInstance(self.x.__str__(), str)

    def test_empty_comment(self):
        self.assertSequenceEqual(self.x.comment(), '')

    def test_add_comment(self):
        self.assertSequenceEqual(self.x.comment(), '')

        s = 'This is a comment\nwith a line break'
        self.x.comment(comment=s)
        print(self.x)

        self.assertSequenceEqual(self.x.comment(), s)

class Test_ConstructorChannels(unittest.TestCase):
    def setUp(self):
        self.x = Audio(channels=4)
        print(self.x)

    def test_str_method(self):
        self.assertIsInstance(self.x.__str__(), str)

    def test_channels(self):
        self.assertEqual(self.x.ch, 4)
        self.assertEqual(len(self.x), 0)

    def test_RMS_is_nan(self):
        print(self.x.rms())
        self.assertTrue(np.isnan(self.x.rms()).all())

    def test_peak_is_nan(self):
        peak, idx = self.x.peak()
        print(peak)
        print(idx)
        self.assertTrue(np.isnan(peak).all())
        self.assertTrue((idx==0).all())

    def test_crestfactor_is_nan(self):
        print(self.x.crest_factor())
        self.assertTrue(np.isnan(self.x.crest_factor()).all())

class Test_ConstructorDuration(unittest.TestCase):
    def test_set_samples(self):
        x = Audio(nofsamples=300, fs=600)
        print(x)
        self.assertAlmostEqual(x.duration, 0.5, places=7)

    def test_set_duration(self):
        x = Audio(duration=1.5, fs=600)
        print(x)
        self.assertEqual(len(x), 900)

    def test_set_duration_and_channels(self):
        x = Audio(duration=1.5, fs=600, channels=5)
        print(x)
        self.assertEqual(len(x), 900)

    def test_set_duration_and_samples(self):
        self.assertRaises(AssertionError, callableObj=Audio, nofsamples=10, duration=1.1)

if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s "+
                                 "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
