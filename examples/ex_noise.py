'''
Created on 16 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Internal
from zignal.audio import Audio, Noise

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.INFO)

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
