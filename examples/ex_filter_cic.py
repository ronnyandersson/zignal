'''
Created on 9 Apr 2020

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2020 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Third party
import matplotlib.pyplot as plt
import numpy as np

# Internal
import zignal

if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
        level='DEBUG')
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Create an FIR filter with n coefficients set to 1/n
    # Plot this and compare it with a cic filter with D=n

    n = 10

    B = np.ones(n)
    B = B / len(B)

    f = zignal.filters.linearfilter.FIR(B=B, fs=1)
    print(f)

    #f.plot_impulse_resp()
    f.plot_pole_zero()
    #f.plot_mag_phase(unwrap=True)

    plt.figure(1)

    plt.plot(*zignal.filters.cic.cic(1, n),
             ls="-",  lw=2.0, c="b", label="CIC 1st order")

    plt.plot(*f.magnitude_resp(frequencies=50000),
             ls="--", lw=2.0, c="r", label="FIR comb")

    plt.ylim(-80, 3)
    plt.xlabel("Normalised frequency")
    plt.ylabel("Magnitude [dB]")
    plt.legend()
    plt.grid()
    plt.show()

    print('++ End of script ++')
