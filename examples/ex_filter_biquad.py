'''
Created on 23 Aug 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Internal
from zignal.filters.biquads import RBJ, Zolzer

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    gaindb  = -6
    f0      = 997
    Q       = 0.707
    fs      = 48000

    # Create a Robert Bristow-Johnson type biquad filter
    f1 = RBJ(filtertype=RBJ.Types.peak, gaindb=gaindb, f0=f0, Q=Q, fs=fs)

    f1.plot_mag_phase()
    f1.plot_pole_zero()

    # Create an Udo Zolzer type biquad filter
    f2 = Zolzer(filtertype=Zolzer.Types.peak, gaindb=gaindb, f0=f0, Q=Q, fs=fs)

    f2.plot_mag_phase()
    f2.plot_pole_zero()
