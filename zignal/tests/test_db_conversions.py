'''
Created on 23 Oct 2014

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
from zignal import db2lin, db2pow, lin2db, pow2db


class Test_back_to_back(unittest.TestCase):
    def test_nested(self):
        x = lin2db(db2pow(pow2db(db2lin(1.234567))))
        self.assertAlmostEqual(x,       1.234567,  places=6)

    def test_lin_to_db_to_lin(self):
        x = lin2db(db2lin(              1.234567))                              # noqa: E201
        self.assertAlmostEqual(x,       1.234567,  places=6)

    def test_pow_to_db_to_pow(self):
        x = pow2db(db2pow(              1.234567))                              # noqa: E201
        self.assertAlmostEqual(x,       1.234567,  places=6)

    def test_lin_to_db_to_lin_arrays(self):
        x = lin2db(db2lin((             1.234567,   2.345678)))                 # noqa: E201
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 2)
        self.assertAlmostEqual(x[0],    1.234567,               places=6)
        self.assertAlmostEqual(x[1],                2.345678,   places=6)

    def test_pow_to_db_to_pow_arrays(self):
        x = pow2db(db2pow((             1.234567,   2.345678)))                 # noqa: E201
        self.assertEqual(x.ndim, 1)
        self.assertEqual(len(x), 2)
        self.assertAlmostEqual(x[0],    1.234567,               places=6)
        self.assertAlmostEqual(x[1],                2.345678,   places=6)


class Test_lin_to_db_known_values(unittest.TestCase):
    def test_known_value_1(self):
        x = lin2db(1)
        self.assertAlmostEqual(x,  0.0,  places=6)

    def test_known_value_0_1(self):
        x = lin2db(0.1)
        self.assertAlmostEqual(x, -20.0, places=6)

    def test_known_value_0_5(self):
        x = lin2db(0.5)
        self.assertAlmostEqual(x, -6.02, places=2)

    def test_known_value_2(self):
        x = lin2db(2)
        self.assertAlmostEqual(x,  6.02, places=2)

    def test_known_value_0_0(self):
        x = lin2db(0.0)
        self.assertTrue(np.isneginf(x))


class Test_db_to_lin_known_values(unittest.TestCase):
    def test_known_value_0(self):
        x = db2lin(0)
        self.assertAlmostEqual(x, 1.0, places=6)

    def test_known_value_neg20(self):
        x = db2lin(-20)
        self.assertAlmostEqual(x, 0.1, places=6)

    def test_known_value_neg6(self):
        x = db2lin(-6)  # -6.020599913
        self.assertAlmostEqual(x, 0.5, places=2)

    def test_known_value_6(self):
        x = db2lin(6)
        self.assertAlmostEqual(x, 2.0, places=2)

    def test_known_value_neg_inf(self):
        x = db2lin(float('-inf'))
        self.assertAlmostEqual(x,  0.0, places=6)


class Test_pow_to_db_known_values(unittest.TestCase):
    def test_known_value_1(self):
        x = pow2db(1)
        self.assertAlmostEqual(x,  0.0,  places=6)

    def test_known_value_0_1(self):
        x = pow2db(0.1)
        self.assertAlmostEqual(x, -10.0, places=6)

    def test_known_value_0_5(self):
        x = pow2db(0.5)
        self.assertAlmostEqual(x, -3.01, places=2)

    def test_known_value_2(self):
        x = pow2db(2)
        self.assertAlmostEqual(x,  3.01, places=2)

    def test_known_value_0_0(self):
        x = pow2db(0.0)
        self.assertTrue(np.isneginf(x))


class Test_db_to_pow_known_values(unittest.TestCase):
    def test_known_value_0(self):
        x = db2pow(0)
        self.assertAlmostEqual(x, 1.0, places=6)

    def test_known_value_3(self):
        x = db2pow(3)
        self.assertAlmostEqual(x, 2.0, places=2)

    def test_known_value_6(self):
        x = db2pow(6.02)
        self.assertAlmostEqual(x, 4.0, places=2)

    def test_known_value_neg_3(self):
        x = db2pow(-3)
        self.assertAlmostEqual(x, 0.5, places=2)

    def test_known_value_neg_10(self):
        x = db2pow(-10)
        self.assertAlmostEqual(x, 0.1, places=2)


class Test_lin_to_db_input_datatypes(unittest.TestCase):
    def test_single(self):
        x = lin2db(1)
        self.assertEqual(x.ndim, 0)
        self.assertAlmostEqual(x, 0.0, places=6)

    def test_tuple(self):
        x = lin2db((1, 0.1))
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(x[0],   0.0, places=6)
        self.assertAlmostEqual(x[1], -20.0, places=6)

    def test_list(self):
        x = lin2db([10, 1])
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(x[0],  20.0, places=6)
        self.assertAlmostEqual(x[1],   0.0, places=6)

    def test_np_ndim_1(self):
        x = lin2db(np.ones(10))
        self.assertEqual(x.ndim, 1)
        self.assertTrue((x <  0.0001).all())                    # noqa: E222
        self.assertTrue((x > -0.0001).all())

    def test_np_ndim_2_10x4(self):
        x = lin2db(np.ones((10, 4)))
        self.assertEqual(x.ndim, 2)
        self.assertTrue((x <  0.0001).all())                    # noqa: E222
        self.assertTrue((x > -0.0001).all())

    def test_np_ndim_2_4x10(self):
        x = lin2db(np.ones((4, 10)))
        self.assertEqual(x.ndim, 2)
        self.assertTrue((x <  0.0001).all())                    # noqa: E222
        self.assertTrue((x > -0.0001).all())


class Test_db_to_lin_input_datatypes(unittest.TestCase):
    def test_single(self):
        x = db2lin(0)
        self.assertEqual(x.ndim, 0)
        self.assertAlmostEqual(x, 1.0, places=6)

    def test_tuple(self):
        x = db2lin((40, -40))
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(x[0],  100.0,  places=6)
        self.assertAlmostEqual(x[1],    0.01, places=6)

    def test_list(self):
        x = db2lin([-20, 60])
        self.assertEqual(x.ndim, 1)
        self.assertAlmostEqual(x[0],    0.1, places=6)
        self.assertAlmostEqual(x[1], 1000.0, places=6)

    def test_np_ndim_1(self):
        x = db2lin(np.zeros(10))
        self.assertEqual(x.ndim, 1)
        self.assertTrue((x < 1.0001).all())
        self.assertTrue((x > 0.9999).all())

    def test_np_ndim_2_10x4(self):
        x = db2lin(np.zeros((10, 4)))
        self.assertEqual(x.ndim, 2)
        self.assertTrue((x < 1.0001).all())
        self.assertTrue((x > 0.9999).all())

    def test_np_ndim_2_4x10(self):
        x = db2lin(np.zeros((4, 10)))
        self.assertEqual(x.ndim, 2)
        self.assertTrue((x < 1.0001).all())
        self.assertTrue((x > 0.9999).all())


if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s " +
                "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
