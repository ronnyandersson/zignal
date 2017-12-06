'''
Created on 24 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# standard library
from __future__ import division, print_function
import unittest

# external libraries
import nose

# local libraries
from zignal.music import scales

class Test_midi_scales(unittest.TestCase):
    # Benson, DJ. (2006). Music: A Mathematical Offering. Cambridge University Press.
    # http://homepages.abdn.ac.uk/mth192/pages/html/maths-music.html
    def test_freq2key_quantise(self):
        # 70 466.164
        # 69 440.00
        # 68 415.305
        self.assertAlmostEqual(scales.midi_freq2key(416.4, quantise=True), 68, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(438.0, quantise=True), 69, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(441.0, quantise=True), 69, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(442.0, quantise=True), 69, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(452.1, quantise=True), 69, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(453.1, quantise=True), 70, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(460.0, quantise=True), 70, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(470.0, quantise=True), 70, places=7)
        
    def test_key2freq(self):
        self.assertAlmostEqual(scales.midi_key2freq(69), 440.0,     places=7)
        self.assertAlmostEqual(scales.midi_key2freq(81), 880.0,     places=7)
        self.assertAlmostEqual(scales.midi_key2freq(21), 27.5,      places=7)
        self.assertAlmostEqual(scales.midi_key2freq(43), 97.9989,   places=4)
        
    def test_freq2key(self):
        self.assertAlmostEqual(scales.midi_freq2key(440), 69.0,     places=7)
        self.assertAlmostEqual(scales.midi_freq2key(880), 81.0,     places=7)
        
    def test_key2freq_tuning(self):
        self.assertAlmostEqual(scales.midi_key2freq(69, tuning=450), 450.0, places=7)
        self.assertAlmostEqual(scales.midi_key2freq(81, tuning=450), 900.0, places=7)
        self.assertAlmostEqual(scales.midi_key2freq(21, tuning=400),  25.0, places=7)
        
    def test_freq2key_tuning(self):
        self.assertAlmostEqual(scales.midi_freq2key(450, tuning=450), 69.0, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(900, tuning=450), 81.0, places=7)
        
    def test_back2back_key(self):
        self.assertAlmostEqual(scales.midi_key2freq(scales.midi_freq2key(1234)),    1234,   places=7)
        self.assertAlmostEqual(scales.midi_key2freq(scales.midi_freq2key(45.67)),   45.67,  places=7)
        
    def test_back2back_freq(self):
        self.assertAlmostEqual(scales.midi_freq2key(scales.midi_key2freq(76.543)),  76.543, places=7)
        self.assertAlmostEqual(scales.midi_freq2key(scales.midi_key2freq(124)),     124,    places=7)
        
class Test_piano_note_to_freq(unittest.TestCase):
    def test_octaves(self):
        self.assertAlmostEqual(scales.piano_note2freq('A2'), 110.0,     places=7)
        self.assertAlmostEqual(scales.piano_note2freq('A3'), 220.0,     places=7)
        self.assertAlmostEqual(scales.piano_note2freq('A4'), 440.0,     places=7)
        self.assertAlmostEqual(scales.piano_note2freq('A5'), 880.0,     places=7)
        self.assertAlmostEqual(scales.piano_note2freq('A6'), 1760.0,    places=7)
        
    def test_values(self):
        self.assertAlmostEqual(scales.piano_note2freq('C6'), 1046.50,   places=2)
        self.assertAlmostEqual(scales.piano_note2freq('D1'), 36.7081,   places=4)
        
class Test_piano_freq_to_note(unittest.TestCase):
    def test_values(self):
        self.assertEqual(scales.piano_freq2note(1046.50),   'C6')
        self.assertEqual(scales.piano_freq2note(36.7051),   'D1')
        self.assertEqual(scales.piano_freq2note(440),       'A4')
        
    def test_quantise(self):
        self.assertEqual(scales.piano_freq2note(435.00), 'A4')
        self.assertEqual(scales.piano_freq2note(439.00), 'A4')
        self.assertEqual(scales.piano_freq2note(440.00), 'A4')
        self.assertEqual(scales.piano_freq2note(441.00), 'A4')
        self.assertEqual(scales.piano_freq2note(447.00), 'A4')
        
class Test_piano(unittest.TestCase):
    def test_back2back_key(self):
        self.assertAlmostEqual(scales.piano_key2freq(scales.piano_freq2key(100)),   100,    places=7)
        self.assertAlmostEqual(scales.piano_key2freq(scales.piano_freq2key(32)),    32,     places=7)
        self.assertAlmostEqual(scales.piano_key2freq(scales.piano_freq2key(997)),   997,    places=7)
        self.assertAlmostEqual(scales.piano_key2freq(scales.piano_freq2key(12345)), 12345,  places=7)
        self.assertAlmostEqual(scales.piano_key2freq(scales.piano_freq2key(4.563)), 4.563,  places=7)
        
    def test_back2back_freq(self):
        self.assertAlmostEqual(scales.piano_freq2key(scales.piano_key2freq(10)), 10,        places=7)
        self.assertAlmostEqual(scales.piano_freq2key(scales.piano_key2freq(49)), 49,        places=7)
        self.assertAlmostEqual(scales.piano_freq2key(scales.piano_key2freq(30.3)), 30.3,    places=7)
        
    def test_back2back_freq_quantised(self):
        self.assertAlmostEqual(scales.piano_freq2key(scales.piano_key2freq(10.2),
                                                     quantise=True), 10, places=7)
        self.assertAlmostEqual(scales.piano_freq2key(scales.piano_key2freq(34.678),
                                                     quantise=True), 35, places=7)

if __name__ == "__main__":
    noseargs = [__name__,
                "--verbosity=2",
                "--logging-format=%(asctime)s %(levelname)-8s: %(name)-15s "+
                                 "%(module)-15s %(funcName)-20s %(message)s",
                "--logging-level=DEBUG",
                __file__,
                ]
    nose.run(argv=noseargs)
