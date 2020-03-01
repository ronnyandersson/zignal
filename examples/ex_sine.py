'''
Created on 15 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Internal
from zignal.audio import Sinetone

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    fs  = 1000
    f0  = 10
    dur = 0.1

    x = Sinetone(f0=f0, fs=fs, duration=dur, gaindb=0)
    print(x)
    x.plot(linestyle='--', marker='x', color='r', label='sine at %i Hz' % f0)

    x.set_sample_rate(500)
    print(x)
    x.plot(linestyle='-.', color='k', label='sine at %i Hz' % f0)
