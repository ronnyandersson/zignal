'''
Created on 21 Jun 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# standard library
import unittest

# external libraries
import nose

# local libraries
from zignal.filters import linearfilter

class Test_Filter(unittest.TestCase):
    def verify_coefficients(self, f, B, A):
        for i, v in enumerate(f.get_coefficients()[0]):
            self.assertAlmostEqual(v, B[i], places=7)

        for i, v in enumerate(f.get_coefficients()[1]):
            self.assertAlmostEqual(v, A[i], places=7)

    def test_get_cefficients(self):
        B = (0.1, 0.2, 0.3)
        A = (0.4, 0.5, 0.6)
        f = linearfilter.Filter(B, A, fs=48000)
        print(f)

        self.verify_coefficients(f, B, A)

    def test_get_feed_forward(self):
        B = (0.1, 0.2, 0.3)
        A = (0.4, 0.5, 0.6)
        f = linearfilter.Filter(B, A, fs=48000)
        print(f)

        BB = f.get_feed_forward()

        self.verify_coefficients(f, BB, A)

    def test_get_feed_back(self):
        B = (0.1, 0.2, 0.3)
        A = (0.4, 0.5, 0.6)
        f = linearfilter.Filter(B, A, fs=48000)
        print(f)

        AA = f.get_feed_back()

        self.verify_coefficients(f, B, AA)

    def test_normalise(self):
        B = (0.1, 0.2, 0.3)
        A = (0.4, 0.5, 0.6)
        f = linearfilter.Filter(B, A, fs=48000)
        print(f)

        f.normalise()
        unused_BB, AA = f.get_coefficients()
        self.assertAlmostEqual(AA[0], 1.0, places=7)

    def test_set_cefficients(self):
        B = (0.7, 0.8, 0.9)
        A = (1.1, 1.2, 1.3)
        f = linearfilter.Filter(fs=48000)
        f.set_coefficients(B, A)
        print(f)

        self.verify_coefficients(f, B, A)

    def test_str_method(self):
        f = linearfilter.Filter()
        self.assertIsInstance(f.__str__(), basestring)

class Test_normalised_frequency(unittest.TestCase):
    def test_full_samplerate(self):
        self.assertAlmostEqual(linearfilter.normalised_frequency(f0=44100, fs=44100), 2.0, places=7)

    def test_half_samplerate(self):
        self.assertAlmostEqual(linearfilter.normalised_frequency(f0=1000,  fs=2000),  1.0, places=7)

    def test_quarter_samplerate(self):
        self.assertAlmostEqual(linearfilter.normalised_frequency(f0=24000, fs=96000), 0.5, places=7)

if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s "+
                                 "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
