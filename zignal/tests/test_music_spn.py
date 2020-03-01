'''
Created on 21 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# Standard library
import unittest

# Third party
import nose

# Internal
from zignal.music import spn


class Test_key2index(unittest.TestCase):
    def test_increasing(self):
        self.assertEqual(spn.key2index('A0'),   1)
        self.assertEqual(spn.key2index('A#0'),  2)
        self.assertEqual(spn.key2index('B0'),   3)
        self.assertEqual(spn.key2index('C1'),   4)
        self.assertEqual(spn.key2index('C#1'),  5)
        self.assertEqual(spn.key2index('D1'),   6)
        self.assertEqual(spn.key2index('D#1'),  7)
        self.assertEqual(spn.key2index('E1'),   8)
        self.assertEqual(spn.key2index('F1'),   9)
        self.assertEqual(spn.key2index('F#1'),  10)
        self.assertEqual(spn.key2index('G1'),   11)
        self.assertEqual(spn.key2index('G#1'),  12)
        self.assertEqual(spn.key2index('A1'),   13)
        self.assertEqual(spn.key2index('A#1'),  14)
        self.assertEqual(spn.key2index('B1'),   15)
        self.assertEqual(spn.key2index('C2'),   16)

    def test_flat(self):
        self.assertEqual(spn.key2index('Abbb4'), 46)
        self.assertEqual(spn.key2index('Abb4'),  47)
        self.assertEqual(spn.key2index('Ab4'),   48)

    def test_standard(self):
        self.assertEqual(spn.key2index('A4'),   49)

    def test_sharp(self):
        self.assertEqual(spn.key2index('A#4'),  50)
        self.assertEqual(spn.key2index('Ax4'),  51)
        self.assertEqual(spn.key2index('A#x4'), 52)

    def test_lowest_piano_key(self):
        self.assertEqual(spn.key2index('A0'),   1)

    def test_beyond_lowest_piano_key(self):
        self.assertEqual(spn.key2index('Ab0'),       0)
        self.assertEqual(spn.key2index('Abb0'),     -1)
        self.assertEqual(spn.key2index('C0'),       -8)
        self.assertEqual(spn.key2index('A-1'),      -11)
        self.assertEqual(spn.key2index('Ab-10'),    -10*12)

    def test_regexp_H(self):
        # H is sometimes used instead of B
        self.assertRaises(AssertionError,   spn.key2index, 'H4')

    def test_C_flat(self):
        #C-flat and B-sharp issues
        self.assertEqual(spn.key2index('Cb4'), spn.key2index('B3'))
        self.assertEqual(spn.key2index('B#4'), spn.key2index('C5'))

    def test_regexp_incomplete(self):
        self.assertRaises(ValueError,       spn.key2index, 'B')
        self.assertRaises(ValueError,       spn.key2index, '7')
        self.assertRaises(ValueError,       spn.key2index, '#')
        self.assertRaises(ValueError,       spn.key2index, 'x')
        self.assertRaises(ValueError,       spn.key2index, 'bb')

    def test_regexp_invalid(self):
        self.assertRaises(AssertionError,   spn.key2index, 'asdfghj')
        self.assertRaises(AssertionError,   spn.key2index, '213ewq')
        self.assertRaises(AssertionError,   spn.key2index, 'c4')
        self.assertRaises(ValueError,       spn.key2index, 'Cxx4')
        self.assertRaises(ValueError,       spn.key2index, 'C##4')
        self.assertRaises(ValueError,       spn.key2index, 'C###4')
        self.assertRaises(ValueError,       spn.key2index, 'Cbbbb4')

    def test_regexp_empty(self):
        self.assertRaises(ValueError,       spn.key2index, '')
        self.assertRaises(AssertionError,   spn.key2index, ' ')

class Test_index2key(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(spn.index2key(49), 'A4')
        self.assertEqual(spn.index2key(1),  'A0')
        self.assertEqual(spn.index2key(-8), 'C0')
        self.assertEqual(spn.index2key(51), 'B4')
        self.assertEqual(spn.index2key(11), 'G1')

    def test_increasing(self):
        self.assertEqual(spn.index2key(1),  'A0' )
        self.assertEqual(spn.index2key(2),  'A#0')
        self.assertEqual(spn.index2key(3),  'B0' )
        self.assertEqual(spn.index2key(4),  'C1' )
        self.assertEqual(spn.index2key(5),  'C#1')
        self.assertEqual(spn.index2key(6),  'D1' )
        self.assertEqual(spn.index2key(7),  'D#1')
        self.assertEqual(spn.index2key(8),  'E1' )
        self.assertEqual(spn.index2key(9),  'F1' )
        self.assertEqual(spn.index2key(10), 'F#1')
        self.assertEqual(spn.index2key(11), 'G1' )
        self.assertEqual(spn.index2key(12), 'G#1')
        self.assertEqual(spn.index2key(13), 'A1' )
        self.assertEqual(spn.index2key(14), 'A#1')
        self.assertEqual(spn.index2key(15), 'B1' )
        self.assertEqual(spn.index2key(16), 'C2' )

    def test_argument_float(self):
        self.assertRaises(AssertionError,   spn.index2key,   1.0)
        self.assertRaises(AssertionError,   spn.index2key,  -1.0)
        self.assertRaises(AssertionError,   spn.index2key,   2.3)
        self.assertRaises(AssertionError,   spn.index2key,  -0.01)

    def test_argument_string(self):
        self.assertRaises(AssertionError,   spn.index2key,  'A4')
        self.assertRaises(AssertionError,   spn.index2key,  '4')

if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s "+
                                 "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
