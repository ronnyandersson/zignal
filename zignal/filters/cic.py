'''
Created on 8 Apr 2020

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
from zignal import lin2db


def cic(N=1, D=10, f=None):
    """
    Cascaded Integrator-Comb filter, CIC filter.

    N     order
    D     decimation
    f     frequencies to evaluate

    The equivalent first order (moving average) FIR filter has D
    number of coefficients all set to 1/D

    For example, cic with N=1 D=10 would have FIR B coefficients set to

        >>> B = np.ones(10)
        >>> B = B / len(B)
        >>> B
        array([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])

    """

    if f is None:
        f = np.linspace(10**-6, 0.5, num=50000)

    # The theoretical frequency response. Only the magnitude response
    # is calculated here.
    m = np.power(np.absolute(np.sin(np.pi*f*D)/np.sin(np.pi*f)), N)

    # normalise so that range is 0..1
    m = m/(D**N)

    # convert to decibel
    db = lin2db(m)

    return f, db


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
        level='DEBUG',
        )
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    # Demo plot, 1st to 4th order cic filters with decimation set to 10
    plt.figure(1)
    plt.plot(*cic(1, 10), ls="-",  label="1st order")
    plt.plot(*cic(2, 10), ls="--", label="2nd order")
    plt.plot(*cic(3, 10), ls="-.", label="3rd order")
    plt.plot(*cic(4, 10), ls=":",  label="4th order")
    plt.ylim(-80, 3)
    plt.xlabel("Normalised frequency")
    plt.ylabel("Magnitude [dB]")
    plt.legend()
    plt.grid()
    plt.show()

    print('++ End of script ++')
