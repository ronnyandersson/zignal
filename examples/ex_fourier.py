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
from zignal.audio import Audio, FourierSeries

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    dur = 0.1
    f0  = 20

    k   = Audio()
    x0  = FourierSeries(f0=f0, duration=dur, harmonics=0,  gaindb=-15)  # fundamental + 0 harmonics
    x1  = FourierSeries(f0=f0, duration=dur, harmonics=1,  gaindb=-15)  # fundamental + 1 harmonics
    x2  = FourierSeries(f0=f0, duration=dur, harmonics=2,  gaindb=-9)   # fundamental + 2 harmonics
    x3  = FourierSeries(f0=f0, duration=dur, harmonics=3,  gaindb=-9)   # ...
    x4  = FourierSeries(f0=f0, duration=dur, harmonics=4,  gaindb=-3)
    x5  = FourierSeries(f0=f0, duration=dur, harmonics=5,  gaindb=-3)
    x60 = FourierSeries(f0=f0, duration=dur, harmonics=60, gaindb=0)    # fundamental + 60 harmonics

    k.append(x0, x1, x2, x3, x4, x5, x60)
    print(k)
    k.plot(ch='all')
