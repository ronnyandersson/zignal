'''
Created on 31 Oct 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
'''

# Standard library
import unittest

# Third party
import nose
import numpy as np

# Internal
from zignal import Audio


class Test_single_channel(unittest.TestCase):
    def setUp(self):
        self.y = np.zeros(4)

    def check_values(self, values, expected, position):
        x = Audio(fs=10, initialdata=values)

        peak, idx = x.peak()
        self.assertTrue(len(peak) == 1)
        self.assertTrue(len(idx) == 1)

        print("index: %3i  peak: %f" % (idx, peak))
        print(x)

        self.assertAlmostEqual(peak, expected, places=3)
        self.assertEqual(idx, position)

    def test_positive(self):
        self.y[1] =  2.0
        self.y[2] =  2.2    # <-- peak
        self.y[3] = -1.2
        print("init data: %s" % self.y)
        self.check_values(self.y, 2.2, 2)

    def test_negative(self):
        self.y[1] =  2.0
        self.y[2] =  3.19
        self.y[3] = -3.2    # <-- peak
        print("init data: %s" % self.y)
        self.check_values(self.y, -3.2, 3)


class Test_multi_channel(unittest.TestCase):
    def setUp(self):
        self.y = np.zeros((4, 2))

    def check_values(self, values, expected, position):
        x = Audio(fs=10, initialdata=values)

        peak, idx = x.peak()
        self.assertTrue(len(peak) == 2)
        self.assertTrue(len(idx) == 2)

        print("index: %s  peak: %s" % (idx, peak))
        print(x)

        self.assertAlmostEqual(peak[0], expected[0], places=3)
        self.assertAlmostEqual(peak[1], expected[1], places=3)

        self.assertEqual(idx[0], position[0])
        self.assertEqual(idx[1], position[1])

    def test_positive(self):
        self.y[1][0] =  1.0
        self.y[2][0] =  2.3     # <-- peak

        self.y[1][1] = -4.1     # <-- peak
        self.y[2][1] =  3.0
        print(self.y)
        self.check_values(self.y, [2.3, -4.1], [2, 1])

    def test_negative(self):
        self.y[1][0] =  1.0
        self.y[2][0] =  2.0     # <-- peak

        self.y[0][1] = -4.0     # <-- peak
        self.y[1][1] =  3.0
        print(self.y)
        self.check_values(self.y, [2, -4], [2, 0])


if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s " +
                "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
