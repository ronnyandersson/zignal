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
from zignal.audio import Noise, Audio

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    fs  = 96000
    dur = 1.0

    nw  = Noise(fs=fs, duration=dur, gaindb=-6, colour='white')
    np  = Noise(fs=fs, duration=dur, gaindb=-6, colour='pink')
    x   = Audio(fs=fs)
    x.append(nw, np)

    print(nw)
    print(np)
    print(x)
    x.plot_fft()
