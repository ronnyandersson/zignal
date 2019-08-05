"""
Created on 16 Feb 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
"""

# standard library
import logging
from abc import ABCMeta, abstractmethod

# external libraries
import numpy as np

# local libraries
from zignal.filters.linearfilter import IIR

class Biquad(IIR):
    """Implements a two-pole, two-zeros biquadratic filter, biquad."""
    def __init__(self, B=None, A=None, fs=96000):
        IIR.__init__(self, B=B, A=A, fs=fs)

    def set_coefficients(self, B=None, A=None):
        """Overrides the method in the parent class so we can assure
        that we always have three B and three A coefficients.
        """
        self._logger.debug("set coefficients (class Biquad)")

        self._B = np.array((1,0,0)) if B is None else np.array(B)
        self._A = np.array((1,0,0)) if A is None else np.array(A)

        assert len(self._B) == 3, "Biquads have three B coefficients"
        assert len(self._A) == 3, "Biquads have three A coefficients"

class BiquadNormalised(Biquad):
    """The normalised biquad always has A[0]=1.0

    This particular implementation keeps the number of coefficients to six in total,
    making sure that the normalise() method is called whenever we calculate the
    coefficients.
    """
    def __init__(self, B=None, A=None, fs=96000):
        Biquad.__init__(self, B=B, A=A, fs=fs)
        self.normalise()

    def set_coefficients(self, B=None, A=None):
        self._logger.debug("set coefficients (class BiquadNormalised)")
        Biquad.set_coefficients(self, B=B, A=A)
        self.normalise()

    def get_coefficients_Pd(self):
        """Return coefficients compatible with the [biquad~] object in Pd"""
        B, A = self.get_coefficients()
        return (B[0], B[1], B[2], -A[1], -A[2]) #FIXME: Verify this

    def get_coefficients_MaxMSP(self):
        """Return coefficients compatible with the [biquad~] object in Max/MSP"""
        B, A = self.get_coefficients()
        return (B[0], B[1], B[2], A[1], A[2])   #FIXME: Verify this

class _BiquadParametric(BiquadNormalised, metaclass=ABCMeta):
    def __init__(self, filtertype=None, gaindb=0, f0=997, Q=0.707, fs=96000):
        BiquadNormalised.__init__(self, B=None, A=None, fs=fs)
        self._verify_parameters(filtertype, gaindb, f0, Q)
        self.calculate_coefficients(filtertype=filtertype, gaindb=gaindb, f0=f0, Q=Q)

    def _verify_parameters(self, filtertype, gaindb, f0, Q):
        """Internal verification that we are at least partially sane before
        proceeding.
        """
        assert filtertype is not None, "Specify a filter type (lowpass, highpass, peak, ..."
        assert f0 >= 0,             "negative frequency is not allowed"
        assert f0 < self.fs/2,      "f0 must be below the Nyquist frequency (fs/2)"
        assert Q > 0,               "Q needs to be positive and above zero (we divide by Q)"

        self.filtertype = filtertype
        self.gaindb     = gaindb
        self.f0         = f0
        self.Q          = Q

    def __str__(self):
        s  = BiquadNormalised.__str__(self)
        # += '-----------------:---------------------\n'
        s += 'type             : %s\n'          %self.filtertype
        s += 'gain             : %.2f [dB]\n'   %self.gaindb
        s += 'f0               : %.1f [Hz]\n'   %self.f0
        s += 'Q                : %.4f\n'        %self.Q
        return s

    @abstractmethod
    def calculate_coefficients(self, filtertype=None, gaindb=None, f0=None, Q=None):
        # Implement this method in child class
        pass

class RBJ(_BiquadParametric):
    """An implementation of the Audio EQ cookbook by Robert Bristow-Johnson

    Robert Bristow-Johnson - Cookbook formulae for audio EQ biquad filter coefficients
    http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
    """

    class Types(object):
        lowpass     = 'lowpass'
        highpass    = 'highpass'
        bandpass1   = 'bandpass1'
        bandpass2   = 'bandpass2'
        notch       = 'notch'
        allpass     = 'allpass'
        peak        = 'peak'
        lowshelf    = 'lowshelf'
        highshelf   = 'highshelf'

    def __init__(self, filtertype=None, gaindb=0, f0=997, Q=0.707, fs=96000):
        _BiquadParametric.__init__(self, filtertype=filtertype, gaindb=gaindb, f0=f0, Q=Q, fs=fs)

    def calculate_coefficients(self, filtertype=None, gaindb=None, f0=None, Q=None):
        self._verify_parameters(filtertype, gaindb, f0, Q)

        # intermediate variables
        _A      = 10**(gaindb/40)
        _w0     = 2 * np.pi * f0/self.fs
        _cos_w0 = np.cos(_w0)
        _sin_w0 = np.sin(_w0)
        _alpha  = _sin_w0/(2*Q)

        if filtertype == self.Types.lowpass:
            # LPF:        H(s) = 1 / (s^2 + s/Q + 1)
            b0 =  (1 - _cos_w0)/2
            b1 =   1 - _cos_w0
            b2 =  (1 - _cos_w0)/2
            a0 =   1 + _alpha
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha

        elif filtertype == self.Types.highpass:
            # HPF:        H(s) = s^2 / (s^2 + s/Q + 1)
            b0 =  (1 + _cos_w0)/2
            b1 = -(1 + _cos_w0)
            b2 =  (1 + _cos_w0)/2
            a0 =   1 + _alpha
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha

        elif filtertype == self.Types.bandpass1:
            # BPF:        H(s) = s / (s^2 + s/Q + 1)  (constant skirt gain, peak gain = Q)
            b0 =   _sin_w0/2    #  =   Q*_alpha
            b1 =   0
            b2 =  -_sin_w0/2    # =  -Q*_alpha
            a0 =   1 + _alpha
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha

        elif filtertype == self.Types.bandpass2:
            # BPF:        H(s) = (s/Q) / (s^2 + s/Q + 1)      (constant 0 dB peak gain)
            b0 =   _alpha
            b1 =   0
            b2 =  -_alpha
            a0 =   1 + _alpha
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha

        elif filtertype == self.Types.notch:
            # notch:      H(s) = (s^2 + 1) / (s^2 + s/Q + 1)
            b0 =   1
            b1 =  -2 * _cos_w0
            b2 =   1
            a0 =   1 + _alpha
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha

        elif filtertype == self.Types.allpass:
            # APF:        H(s) = (s^2 - s/Q + 1) / (s^2 + s/Q + 1)
            b0 =   1 - _alpha
            b1 =  -2 * _cos_w0
            b2 =   1 + _alpha
            a0 =   1 + _alpha
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha

        elif filtertype == self.Types.peak:
            # peakingEQ:  H(s) = (s^2 + s*(A/Q) + 1) / (s^2 + s/(A*Q) + 1)
            b0 =   1 + _alpha*_A
            b1 =  -2 * _cos_w0
            b2 =   1 - _alpha*_A
            a0 =   1 + _alpha/_A
            a1 =  -2 * _cos_w0
            a2 =   1 - _alpha/_A

        elif filtertype == self.Types.lowshelf:
            # lowShelf: H(s) = A * (s^2 + (sqrt(A)/Q)*s + A)/(A*s^2 + (sqrt(A)/Q)*s + 1)
            b0 =    _A*( (_A+1) - (_A-1)*_cos_w0 + 2*np.sqrt(_A)*_alpha )
            b1 =  2*_A*( (_A-1) - (_A+1)*_cos_w0                        )
            b2 =    _A*( (_A+1) - (_A-1)*_cos_w0 - 2*np.sqrt(_A)*_alpha )
            a0 =         (_A+1) + (_A-1)*_cos_w0 + 2*np.sqrt(_A)*_alpha
            a1 =    -2*( (_A-1) + (_A+1)*_cos_w0                        )
            a2 =         (_A+1) + (_A-1)*_cos_w0 - 2*np.sqrt(_A)*_alpha

        elif filtertype == self.Types.highshelf:
            # highShelf: H(s) = A * (A*s^2 + (sqrt(A)/Q)*s + 1)/(s^2 + (sqrt(A)/Q)*s + A)
            b0 =    _A*( (_A+1) + (_A-1)*_cos_w0 + 2*np.sqrt(_A)*_alpha )
            b1 = -2*_A*( (_A-1) + (_A+1)*_cos_w0                        )
            b2 =    _A*( (_A+1) + (_A-1)*_cos_w0 - 2*np.sqrt(_A)*_alpha )
            a0 =         (_A+1) - (_A-1)*_cos_w0 + 2*np.sqrt(_A)*_alpha
            a1 =     2*( (_A-1) - (_A+1)*_cos_w0                        )
            a2 =         (_A+1) - (_A-1)*_cos_w0 - 2*np.sqrt(_A)*_alpha

        else:
            valid = [i for i in vars(self.Types) if not i.startswith("__")]
            raise NotImplementedError("Valid types are: %s" %valid)

        self.set_coefficients(B=(b0, b1, b2), A=(a0, a1, a2))

class Zolzer(_BiquadParametric):
    """An implementation of Equalizer filters from DAFX - Zolzer et al."""
    class Types(object):
        lowpass     = 'lowpass'
        highpass    = 'highpass'
        peak        = 'peak'
        lowshelf    = 'lowshelf'
        highshelf   = 'highshelf'

    def __init__(self, filtertype=None, gaindb=0, f0=997, Q=0.707, fs=96000):
        _BiquadParametric.__init__(self, filtertype=filtertype, gaindb=gaindb, f0=f0, Q=Q, fs=fs)

    def calculate_coefficients(self, filtertype=None, gaindb=None, f0=None, Q=None):
        self._verify_parameters(filtertype, gaindb, f0, Q)

        K = np.tan(np.pi*f0/self.fs)

        #------------
        # peak
        #------------
        if filtertype == self.Types.peak:
            if gaindb > 0:
                self._logger.debug('peak boost')
                V0  = 10**(gaindb/20)
                den =  1 +  1/Q * K + K**2

                b0  = (1 + V0/Q * K + K**2)         / den
                b1  = (2 * (K**2 - 1))              / den
                b2  = (1 - V0/Q * K + K**2)         / den
                a1  = b1
                a2  = (1 -  1/Q * K + K**2)         / den
            else:
                self._logger.debug('peak cut')
                V0  = 10**(-gaindb/20)
                den =  1 + V0/Q * K + K**2
                b0  = (1 +  1/Q * K + K**2)         / den
                b1  = (          2 * (K**2 - 1))    / den
                b2  = (1 -  1/Q * K + K**2)         / den
                a1  = b1
                a2  = (1 - V0/Q * K + K**2)         / den

        #------------
        # low shelf
        #------------
        elif filtertype == self.Types.lowshelf:
            # Parameter 'Q' is not used
            if gaindb > 0:
                self._logger.debug('lowshelf boost')
                V0  = 10**(gaindb/20)
                den =  1+np.sqrt(2)*K + K**2
                b0  = (1+np.sqrt(V0*2)*K + V0*K**2) / den
                b1  = (2*(V0*K**2-1))               / den
                b2  = (1-np.sqrt(V0*2)*K + V0*K**2) / den
                a1  = (2*(K**2-1))                  / den
                a2  = (1-np.sqrt(2)*K + K**2)       / den
            else:
                self._logger.debug('lowshelf cut')
                V0  = 10**(-gaindb/20)
                den =  1+np.sqrt(2*V0)*K + V0*K**2
                b0  = (1+np.sqrt(2)*K + K**2)       / den
                b1  = (2*(K**2-1))                  / den
                b2  = (1-np.sqrt(2)*K + K**2)       / den
                a1  = (2*(V0*K**2-1))               / den
                a2  = (1-np.sqrt(2*V0)*K + V0*K**2) / den

        #------------
        # high shelf
        #------------
        elif filtertype == self.Types.highshelf:
            # Parameter 'Q' is not used
            if gaindb > 0:
                self._logger.debug('highshelf boost')
                V0  = 10**(gaindb/20)
                den =  1+np.sqrt(2)*K + K**2
                b0  = (V0+np.sqrt(V0*2)*K + K**2)   / den
                b1  = (2*(K**2-V0))                 / den
                b2  = (V0-np.sqrt(V0*2)*K + K**2)   / den
                a1  = (2*(K**2-1))                  / den
                a2  = (1-np.sqrt(2)*K + K**2)       / den
            else:
                self._logger.debug('highshelf cut')
                V0  = 10**(-gaindb/20)
                b0  = (1+np.sqrt(2)*K + K**2)           / (V0+np.sqrt(2*V0)*K +  K**2)
                b1  = (2*(K**2-1))                      / (V0+np.sqrt(2*V0)*K +  K**2)
                b2  = (1-np.sqrt(2)*K + K**2)           / (V0+np.sqrt(2*V0)*K +  K**2)
                a1  = (2*(((K**2)/V0)-1))               / ( 1+np.sqrt(2/V0)*K + (K**2)/V0)
                a2  = (1-np.sqrt(2/V0)*K + (K**2)/V0)   / ( 1+np.sqrt(2/V0)*K + (K**2)/V0)

        #------------
        # low pass
        #------------
        elif filtertype == self.Types.lowpass:
            # Parameter 'Q' is not used
            self._logger.debug('lowpass')

            den =  1+np.sqrt(2)*K + K**2
            b0  = (K**2)                    / den
            b1  = (2*K**2)                  / den
            b2  = (K**2)                    / den
            a1  = (2*(K**2-1))              / den
            a2  = (1-np.sqrt(2)*K + K**2)   / den

        #------------
        # high pass
        #------------
        elif filtertype == self.Types.highpass:
            # Parameter 'Q' is not used
            self._logger.debug('highpass')

            den =  1+np.sqrt(2)*K + K**2
            b0  = (1)                       / den
            b1  = (-2)                      / den
            b2  = (1)                       / den
            a1  = (2*(K**2-1))              / den
            a2  = (1-np.sqrt(2)*K + K**2)   / den

        else:
            raise NotImplementedError()

        self.set_coefficients(B=(b0, b1, b2), A=(1.0, a1, a2))

__all__ = ['Biquad', 'RBJ', 'Zolzer',]

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    print('++ End of script ++')
