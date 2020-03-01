'''
Created on 15 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Internal
from zignal.audio import Noise, Sinetone

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    fs  = 96000
    f0  = 997
    dur = 2.5

    x   = Sinetone(f0=f0, fs=fs, duration=dur, gaindb=0)
    n   = Noise(channels=1, fs=fs, duration=dur, gaindb=-50)
    print(x)
    print(n)

    x.samples = x.samples+n.samples

    x.plot_fft(window='hamming')
