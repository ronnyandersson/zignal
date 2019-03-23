'''
Created on 16 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# standard library
from __future__ import print_function
import logging

# custom libraries
from zignal.audio import Sinetone, Noise, Audio, SquareWave

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    fs  = 48000
    dur = 1.5

    x1 = Sinetone(f0=997, fs=fs, duration=dur, gaindb=0)
    x2 = Noise(fs=fs, duration=dur, gaindb=-6)
    x3 = SquareWave(f0=3000, fs=fs, duration=dur, gaindb=-20)
    x4 = Audio(fs=fs)
    x4.append(x1, x2, x3)

    print(x1)
    print(x1.peak())
    print(x1.rms())
    print(x1.crest_factor())

    print(x2)
    print(x2.peak())
    print(x2.rms())
    print(x2.crest_factor())

    print(x3)
    print(x3.peak())
    print(x3.rms())
    print(x3.crest_factor())

    print(x4)
    print(x4.peak())
    print(x4.rms())
    print(x4.crest_factor())
