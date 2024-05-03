'''
Created on 28 Feb 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
'''

# Standard library
import logging
import unittest

# Third party
import numpy as np

# Internal
from zignal import Audio


class Test_EmptyConstructor(unittest.TestCase):
    def setUp(self):
        self.x = Audio()

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
        self.assertSequenceEqual(self.x.comment(), s, msg="\n"+str(self.x))


class Test_ConstructorChannels(unittest.TestCase):
    def setUp(self):
        self.x = Audio(channels=4)

    def test_str_method(self):
        self.assertIsInstance(self.x.__str__(), str)

    def test_channels(self):
        self.assertEqual(self.x.ch, 4)
        self.assertEqual(len(self.x), 0)

    def test_RMS_is_nan(self):
        self.assertTrue(np.isnan(self.x.rms()).all(), msg=self.x.rms())

    def test_peak_is_nan(self):
        peak, idx = self.x.peak()
        self.assertTrue(np.isnan(peak).all(), msg=str(peak))
        self.assertTrue((idx == 0).all(), msg=str(idx))

    def test_crestfactor_is_nan(self):
        self.assertTrue(np.isnan(self.x.crest_factor()).all(), msg=self.x.crest_factor())


class Test_ConstructorDuration(unittest.TestCase):
    def test_set_samples(self):
        x = Audio(nofsamples=300, fs=600)
        self.assertAlmostEqual(x.duration, 0.5, places=7, msg="\n"+str(x))

    def test_set_duration(self):
        x = Audio(duration=1.5, fs=600)
        self.assertEqual(len(x), 900, msg="\n"+str(x))

    def test_set_duration_and_channels(self):
        x = Audio(duration=1.5, fs=600, channels=5)
        self.assertEqual(len(x), 900, msg="\n"+str(x))

    def test_set_duration_and_samples(self):
        self.assertRaises(AssertionError, Audio, nofsamples=10, duration=1.1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)-8s: %(module)s.%(funcName)-15s %(message)s",
        level="DEBUG",
        )
    # some libraries are noisy in DEBUG
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # from command line:
    # $ python -m unittest -v zignal/tests/<filename>.py
    unittest.main(verbosity=2)
