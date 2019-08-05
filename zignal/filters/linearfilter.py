"""
Created on 25 Jan 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
"""

# standard library
import logging

# external libraries
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# local libraries
from zignal import hz2rad, rad2hz

#===================================================================================================
# Linear filter implementations
#
# Julius O. Smith III - Introduction to Digital Filters with audio applications
# https://ccrma.stanford.edu/~jos/fp/
#
#===================================================================================================
class Filter(object):
    def __init__(self, B=None, A=None, fs=96000):
        """Standard linear recursive filter

                b0 + b1*z^(-1) + b2*z^(-2) + ... + bn*z^(-n)
        H(z) = ------------------------
                a0 + a1*z^(-1) + a2*z^(-2) + ... + an*z^(-n)

        B is feed forward     (numerator)    zeros
        A is feed back        (denominator)  poles
        """
        self._logger = logging.getLogger(__name__)
        assert fs > 0
        self.fs = fs
        self._B = None
        self._A = None

        self.set_coefficients(B=B, A=A)

    def __str__(self):
        s  = '=======================================\n'
        s += 'classname        : %s\n'          %self.__class__.__name__
        s += 'sample rate      : %.1f [Hz]\n'   %self.fs
        s += 'feedforward  (B) : %s\n'          %str(self._B)
        s += 'feedback     (A) : %s\n'          %str(self._A)
        s += 'number of zeros  : %i\n'          %(len(self._B)-1)
        s += 'number of poles  : %i\n'          %(len(self._A)-1)
        s += 'minimum phase?   : %s\n'          %("Yes" if self.is_minimum_phase() else "No")
        s += '-----------------:---------------------\n'
        return s

    def __repr__(self):
        return "Filter(B=%s, A=%s, fs=%s)" %(list(self._B), list(self._A), self.fs)

    def filter_samples(self, samples):
        return scipy.signal.lfilter(self._B, self._A, samples, axis=0)

    def set_coefficients(self, B=None, A=None):
        """Set the filter coefficients.

        B is coefficients in the numerator. This determines the location of the zeros.
        B is also called feedforward coefficients. An FIR filter only has B coefficients
        and will always be stable.

        A is coefficients in the denominator. This determines the location of the poles.
        A is also called feedback coefficients. An IIR filter must have A coefficients
        and might become unstable.
        """

        self._logger.debug("set coefficients (class Filter)")

        # numerator (zeros, feedforward, FIR)
        self._B = np.array((1,)) if B is None else np.array(B)

        # denominator (poles, feedback, IIR)
        self._A = np.array((1,)) if A is None else np.array(A)

        assert len(self._B) != 0
        assert len(self._A) != 0

    def get_coefficients(self):
        return self._B, self._A

    def get_feed_forward(self):
        return self._B

    def get_feed_back(self):
        return self._A

    def normalise(self):
        """Normalise the coefficients by dividing by A[0], effectively
        setting A[0]=1.0
        """
        assert len(self._A) >= 1
        assert len(self._B) >= 1

        a0 = self._A[0]
        self._logger.debug("normalising using a0: %.4f" %a0)

        self._B = self._B/a0
        self._A = self._A/a0

    def is_stable(self):
        """A filter is stable if all its poles are strictly inside the unit circle. A
        pole on the unit circle may be called marginally stable.

        TODO: calculate the reflection coefficients instead, since higher order filters
        are sensitive to coefficient rounding errors in the root finding procedure.
        """
        isStable = True

        unused_zeros, poles, unused_gain = scipy.signal.tf2zpk(self._B, self._A)

        for pole in poles:
            if not np.abs(pole) < 1.0:
                isStable = False

        return isStable

    def is_minimum_phase(self):
        """A filter is defined as _minimum phase_ if all its poles and zeros are inside
        the unit circle, excluding the unit circle itself.
        """
        isMinPhase = True

        zeros, poles, unused_gain = scipy.signal.tf2zpk(self._B, self._A)

        for pole in poles:
            if not np.abs(pole) < 1.0:
                isMinPhase = False

        for zero in zeros:
            if not np.abs(zero) < 1.0:
                isMinPhase = False

        return isMinPhase

    def complex_freq_resp(self, frequencies=None):
        """Calculate the complex frequency response"""
        if frequencies is None:
            w, h = scipy.signal.freqz(self._B, self._A, worN=None)
        elif isinstance(frequencies, int):
            w, h = scipy.signal.freqz(self._B, self._A, worN=frequencies)
        else:
            w, h = scipy.signal.freqz(self._B, self._A, worN=hz2rad(frequencies, self.fs))

        return w, h

    def magnitude_resp(self, frequencies=None):
        """Calculate the real magnitude (frequency) response"""
        w, h    = self.complex_freq_resp(frequencies)
        mag     = 20*np.log10(np.absolute(h))
        freqs   = rad2hz(w, self.fs)

        return freqs, mag

    def phase_resp(self, frequencies=None, unwrap=False):
        """Calculate the real phase response"""
        w, h    = self.complex_freq_resp(frequencies)
        phase   = np.angle(h, deg=False)
        phase   = np.unwrap(phase) if unwrap else phase
        phase   = np.rad2deg(phase)
        freqs   = rad2hz(w, self.fs)

        return freqs, phase

    def plot_mag_phase(self, filename=None, plotpoints=10000, unwrap=False):
        """Produce a plot with magnitude and phase response in the same
        figure. The y-axis on the left side belongs to the magnitude
        response. The y-axis on the right side belongs to the phase
        response
        """
        unused_freq, mag = self.magnitude_resp(plotpoints)
        freq, pha = self.phase_resp(plotpoints, unwrap=unwrap)

        fig = plt.figure(1)
        ax_mag = fig.add_subplot(111)
        ax_pha = ax_mag.twinx()
        ax_mag.semilogx(freq, mag, label='magnitude', color='red',  linestyle='-')
        ax_pha.semilogx(freq, pha, label='phase',     color='blue', linestyle='--')
        ax_mag.grid(True)

        ax_mag.set_xlim(10, self.fs/2)
        #ax_mag.set_ylim(bottom=-80)    # FIXME: ad proper padding
        #ax_mag.margins(0.1)

        ax_mag.set_title('Frequency response')
        ax_mag.set_xlabel('Frequency [Hz]')
        ax_mag.set_ylabel('Magnitude [dB]')
        ax_pha.set_ylabel('Phase [deg]')

        handles1, labels1 = ax_mag.get_legend_handles_labels()
        handles2, labels2 = ax_pha.get_legend_handles_labels()

        plt.legend(handles1 + handles2, labels1 + labels2, loc='best')

        if filename is None:
            plt.show()
        else:
            try:
                plt.savefig(filename)
            finally:
                plt.close(1)

    def plot_pole_zero(self, filename=None):
        """Produce a plot with the location of all poles and zeros."""
        zeros, poles, gain = scipy.signal.tf2zpk(self._B, self._A)
        self._logger.debug("zeros: %s" %zeros)
        self._logger.debug("poles: %s" %poles)
        self._logger.debug("gain : %s" %gain)

        fig = plt.figure(1)
        ax = fig.add_subplot(111, aspect='equal')
        circ = plt.Circle((0,0), radius=1, fill=False, color='black',
                          linestyle='dashed', linewidth=1.0)
        ax.add_patch(circ)
        ax.axhline(0, linestyle='dashed', color='black', linewidth=1.0)
        ax.axvline(0, linestyle='dashed', color='black', linewidth=1.0)
        ax.grid(True)

        ax.plot(poles.real, poles.imag, marker='x', ms=7.0, mew=1.5, mfc='blue', mec='blue',
                ls='None', label='poles (%i)' %len(poles))
        ax.plot(zeros.real, zeros.imag, marker='o', ms=7.0, mew=1.5, mfc='None', mec='red',
                ls='None', label='zeros (%i)' %len(zeros))

        ax.margins(0.1)

        # TODO: count multiples at (0,0)

        plt.legend(loc='best', numpoints=1)
        plt.title('Pole-zero locations')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')

        if filename is None:
            plt.show()
        else:
            try:
                plt.savefig(filename)
            finally:
                plt.close(1)

    def plot_impulse_resp(self, filename=None, points=1000):
        """Plot the impulse response"""
        t, m = scipy.signal.dimpulse((self._B, self._A, 1/self.fs), n=points)
        m = m[0]

        #assert isinstance(m, np.ndarray)

        zoom = np.max(np.abs(m[1:]))

        plt.figure(1)
        plt.plot(t, m, '-x', label='impulse response')
        ax = plt.axis()
        plt.axis([ax[0], ax[1], -zoom*1.25, zoom*1.25])
        plt.grid(True)
        plt.title('Impulse response')
        plt.xlabel('Time [s]')
        plt.ylabel('Magnitude')

        if filename is None:
            plt.show()
        else:
            try:
                plt.savefig(filename)
            finally:
                plt.close(1)

class FIR(Filter):
    """Finite Impulse Response filter

    Also known as
     * nonrecursive filter
     * tapped delay line filter
     * moving average filter
     * transversal filter

    Per definition always stable (no feedback coefficients)
    """
    def __init__(self, B=None, fs=96000):
        Filter.__init__(self, B=B, A=(1,), fs=fs)

    def __str__(self):
        s  = Filter.__str__(self)
        # += '-----------------:---------------------\n'
        s += 'noise amplf.     : %s\n' %self.noise_amplification()
        return s

    def noise_amplification(self):
        '''The noise amplification is the sum of the squares of the coefficients'''
        return np.sum(np.power(self._B, 2))

class IIR(Filter):
    """Infinite Impulse Response

    Also known as
     * recursive filter
     * Ladder filter
     * Lattice filter
     * autoregressive moving average filter (ARMA)
     * autoregressive integrated moving average filter (ARIMA)

    """
    def __init__(self, B=None, A=None, fs=96000):
        Filter.__init__(self, B=B, A=A, fs=fs)

    def __str__(self):
        s  = Filter.__str__(self)
        # += '-----------------:---------------------\n'
        s += 'stable?          : %s\n' %("Yes" if self.is_stable() else "No")
        return s

#===================================================================================================
# Functions
#===================================================================================================

def normalised_frequency(f0=1000, fs=96000):
    '''Calculate a normalised frequency between [0.0, 1.0] where 1.0
    corresponds to pi [rad/sample]
    '''
    return f0/(fs/2)

__all__ = ['Filter', 'FIR', 'IIR', 'normalised_frequency',]

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    print('++ End of script ++')
