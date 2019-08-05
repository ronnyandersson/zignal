"""
Created on 22 Mar 2014

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2014 Ronny Andersson
@license: MIT
"""

# standard library
from collections import deque
import logging
import random

# external libraries
import numpy as np
import scipy.signal

# local libraries
from zignal import Audio
from zignal.filters.biquads import RBJ
from zignal.filters.linearfilter import Filter

# expose the measure.mlstaps.TAPS dictionary in the measure.mls namespace
from .mlstaps import *

__all__ = [
           'MLS',
           'MLS_simple',
           'get_random_taps',
           'TAPS',
           ]

def get_random_taps(N):
    '''Select a random set of taps for given order N'''
    return random.choice(TAPS.get(N))

class _MLS_base(object):
    '''Base class for Maximum Length Sequence (MLS) generation. This class
    implements the "tricky" bits when it comes to MLS sequences. An efficient
    generator is needed since MLS is based on shift registers where we are
    dealing with bits.

    It should not be needed to create an instance of this class directly. Use
    the derived child class instead. When debugging or trying to understand
    the inner workings of MLS then this class can be instantiated with a low
    value for N. For example, N=3 will yield an MLS sequence of length 7. This
    is short enough for a pen and paper exercise when trying to apply circular
    cross correlation.

    The basis of MLS theory is not depending on a sample rate. This is because
    we are dealing with sequences of ones and zeros. This can be translated to
    samples with values -1.0 and +1.0 but unaware of any *sample rate*.

    We can later filter the sequence and apply emphasis. At that stage we need
    to know about sample rate but not in this class.

    This class implements:
    * Generators
    * Circular Cross Correlation
    * Extraction of the impulse response
    '''
    def __init__(self, N=None, taps=None):
        assert N    is not None, "Please specify MLS order"
        assert taps is not None, "Please specify feedback taps"
        assert isinstance(taps, (tuple, list))
        assert len(taps) is not 0, "taps are empty!"

        self._logger    = logging.getLogger(__name__)
        self.N          = N
        self.L          = (2**N)-1
        self.taps       = taps

        sizeof_int = np.int64().dtype.itemsize
        self._RAM_usage = (self.L*sizeof_int)/(1024**2) # in Mb

    def __repr__(self):
        return '_MLS_base(N=%i, taps=%s)' %(self.N, tuple(self.taps))

    def __str__(self):
        s  = '=======================================\n'
        s += 'classname        : %s\n'      %self.__class__.__name__
        s += 'N                : %i\n'      %self.N
        s += 'L=(2^N)-1        : %i\n'      %self.L
        s += 'taps             : %s\n'      %str(self.taps)
        s += "RAM              : %.1f [Mb] (one full sequence)\n" %self._RAM_usage
        s += '-----------------:---------------------\n'
        return s

    def generator_bit(self):
        """Generate one bit at a time. This generator never finishes, it will
        generate bits forever. However, the sequence will repeat itself after
        (2^N)-1 steps.

        The returned value is 1 or 0
        """

        # deque is memory efficient and fast. Here we use it as a shift
        # register with a fixed length.
        shiftregister = deque([0]*(self.N-1)+[1], maxlen=self.N)
        #self._logger.debug("*init* %s" %str(shiftregister))

        while True:
            bitvalue = 0
            for tap in self.taps:
                bitvalue = bitvalue ^ shiftregister[self.N-tap]
            shiftregister.append(bitvalue)

            # The genration of one bit every iteration is costly enough. For debugging
            # purposes, uncomment below to get the internal state of the shiftregister
            # for every iteration, given the conditional N < 6.
            #if self.N < 6:
            #    self._logger.debug("state: %s" %str(shiftregister))
            yield bitvalue

    def generator_chunk(self, chunk=1024):
        """Generate a chunk of values, forever. Wraps around at interval (2^N)-1

        Returns a numpy array of length=chunk with datatype integers.
        """
        bitgenerator = self.generator_bit()

        while True:
            seq = np.zeros((chunk, 1), dtype=np.int64)
            for i in range(chunk):
                seq[i] = next(bitgenerator)
            yield seq

    def generator_samples(self, chunk=1024):
        """Generate a chunk of audio samples in the range [-1.0, 1.0]

        Returns a numpy array. The datatype is 64 bit floats.
        """
        chunkgen = self.generator_chunk(chunk)
        while True:
            binarychunk = next(chunkgen)

            # create a view of the data. This points to the same memory structure.
            # Compare this to the union structure in c++. Data size must match that
            # of the generator_chunk() method (sizeof(np.int64)==sizeof(np.float64))
            samples = binarychunk.view(np.float64)

            # The values that the binarychunk can be is 0 and 1. We want this to represent
            # a sample so we need to convert this to -1.0 and +1.0 instead. We do this by
            # multiplying by two and subtracting 1. The values are now converted so that
            # 0 is -1 and +1 is unchanged. Range is now [-1, 1]. Finally we flip the sign
            # of the sequence.

            samples[:] = -(binarychunk*2 - 1)

            yield samples

    def get_full_sequence(self, repeats=1):
        """Get the full MLS sequence as audio samples, repeated n times. When
        extracting the impulse response we need to throw away the first MLS
        sequence. Hence we need to repeat the signal at least once, i.e
        repeat=2. More repetitions will mean longer signal but better signal
        to noise ratio since we will average away any noise.
        """

        chunkgen        = self.generator_samples(chunk=self.L)
        full_sequence   = next(chunkgen)
        reps_sequence   = np.tile(full_sequence.T, repeats).T
        self._logger.debug("May share memory: %s" %np.may_share_memory(full_sequence,
                                                                       reps_sequence))
        return reps_sequence

    def xcorr_circular(self, other):
        """Circular cross correlation. The difference between circular cross
        correlation and regular correlation is that with circular correlation
        we assume that the signal is repeating.

        The input data "other" has to be the same size as a full sequence.

        Returns the (normalised) impulse response of length L.
        """

        #    Example:
        #
        #    _MLS_base(N=3, taps=(3, 2))
        #
        #    reference  : [[ 1. -1. -1. -1.  1.  1. -1.]]
        #
        #    <do cross-correlation>  <-- in this example this is the auto correlation
        #
        #    xcorr      : [[-1.  2.  1. -2. -3.  0.  7.  0. -3. -2.  1.  2. -1.]]
        #    x1 (view)  :                             [[ 0. -3. -2.  1.  2. -1.]]
        #    x2 (view)  : [[-1.  2.  1. -2. -3.  0.]]
        #
        #    <assume sequence is circular>
        #
        #    x1 (view)  : [[ 0. -3. -2.  1.  2. -1.]]
        #    x2 (view)  : [[-1.  2.  1. -2. -3.  0.]]
        #    x2=x1+x2   :   -1. -1. -1. -1. -1. -1.
        #
        #    because we use a "view" of the array
        #
        #    xcorr      : [[-1.  2.  1. -2. -3.  0.  7. -1. -1. -1. -1. -1. -1.]]
        #
        #    norm       :                         [[ 7. -1. -1. -1. -1. -1. -1.]]
        #
        #    <normalise by L>
        #
        #    norm/L     : [[ 1.     -0.1429 -0.1429 -0.1429 -0.1429 -0.1429 -0.1429]]

        ref = self.get_full_sequence(repeats=1)

        # Correlation and convolution are related. Correlation in the time domain is *very*
        # slow for long sequences. Convolution in the frequecy domain is much faster. We can
        # use the convolution method if we "undo" the flip of the input signal. This is
        # equivalent to the correlation method. There might be some small (insignificant)
        # rounding errors but for a large array this should not be noticable.
        #
        ### Flip comments to verify that correlation and convolution (with input flip) is
        ### the same. Be careful, long sequences are slow to calculate using the
        ### correlation method.
        #xcorr   = scipy.signal.correlate(ref, other)
        xcorr = scipy.signal.fftconvolve(np.flipud(ref), other)

        self._logger.debug("ref: %s" %np.array_str(ref.T,     max_line_width=200, precision=4, suppress_small=True))
        self._logger.debug("xc : %s" %np.array_str(xcorr.T,   max_line_width=200, precision=4, suppress_small=True))

        del ref

        # slicing creates views which are cheap (points to the same array)
        x1 = xcorr[self.L:]         # right half (end)
        x2 = xcorr[:self.L-1]       # left halt (start)
        self._logger.debug("x1 : %s" %np.array_str(x1.T,      max_line_width=200, precision=4, suppress_small=True))
        self._logger.debug("x2 : %s" %np.array_str(x2.T,      max_line_width=200, precision=4, suppress_small=True))

        x1[:]=x1+x2                 # assume circular sequence
        self._logger.debug("xc : %s" %np.array_str(xcorr.T,   max_line_width=200, precision=4, suppress_small=True))

        norm = xcorr[self.L-1:]     # extract "impulse" + tail (right half)
        self._logger.debug("nrm: %s" %np.array_str(norm.T,    max_line_width=200, precision=4, suppress_small=True))

        norm[:] = norm/self.L       # normalise so that max <= 1.0
        self._logger.debug("nrm: %s" %np.array_str(norm.T,    max_line_width=200, precision=4, suppress_small=True))

        return norm

    def trim_and_avgerage(self, other):
        """It is assumed that at least two full MLSes are used and sent through the
        system we want to measure. By using m repeated sequences of MLSes we can throw
        away the first full sequence. This first part might be delayed because of
        latency and will contain the startup response of the system which we aren't
        interested in. We then have n=m-1 sequences. Remember that MLS is a repeated
        sequence and it doesn't matter that we have for example latency of for example
        one third of a sequence. This will show up as phase shift which can be
        compensated for.

        We now calculate the average of the n sequences. This gives us even more noise
        resistance.

        Returns a numpy array of length L.
        """
        # data is [[A_1],
        #          [A_2],
        #          [A_3],
        #          [B_1],
        #          [B_2],
        #          [B_3],
        #          [C_1],
        #          [C_2],
        #          [C_3]]

        # throw away first full sequence
        trimmed = other[self.L:]
        self._logger.debug("May share memory: %s" %np.may_share_memory(other, trimmed))
        #print(repr(trimmed))
        # data is [[B_1],
        #          [B_2],
        #          [B_3],
        #          [C_1],
        #          [C_2],
        #          [C_3]]

        repeats = len(trimmed)//self.L
        self._logger.debug("repeats (first discared): %i" %repeats)

        # reshape so we can average
        reshaped = trimmed.reshape((repeats, self.L))
        # data is [[B_1, B_2, B_3],
        #          [C_1, C_2, C_3]]

        self._logger.debug("May share memory: %s" %np.may_share_memory(other, reshaped))
        #print(repr(reshaped))

        # the averaging finally creates a new array that has it's own data
        average = np.average(reshaped, axis=0)

        average = np.expand_dims(average, axis=1) # up the ndim to 2
        # data is [[X_1],
        #          [X_2],
        #          [X_3]]

        return average

    def get_impulse(self, x):
        """Extract the impulse response by averaging the sequences and then
        applying circular cross correlation. The length of input x must be
        a multiple of L.

        Returns a numpy array with the extracted impulse response.
        """
        assert isinstance(x, np.ndarray)
        assert len(x) > self.L, "The first sequence will be thrown away"

        avg     = self.trim_and_avgerage(x)
        impulse = self.xcorr_circular(avg)

        return impulse

class MLS(_MLS_base, Audio):
    """This class is a mixture of the MLS base class and the Audio class. The MLS data
    is the audio samples in the Audio class. Because we also know about sample rate
    here we can apply emphasis and de-emphasis on the samples. When plotting the FFT
    of the impulse response remember to use a rectangular window. This is perfectly
    valid since the MLS is cyclic by nature and will wrap around.

    This class is limited to one channel of audio. Creating a multichannel MLS signal
    is outside the scope of this class. It can easily be done by creating multiple
    instances and then extracting the audio samples and constructing a new Audio
    instance with all channels appended. To extract the impulse response we then need
    to keep track of which channel is paired with which MLS since they should
    preferably be using different taps to minimise cross talk.
    """
    def __init__(self, N=None, taps=None, fs=96000, repeats=2, B=(1, 0, 0), A=(1, 0, 0)):
        """N is the order of the MLS, taps are preferably selected from the mlstaps
        dictionary, like

            >>> taps=TAPS[N][0]

        B and A are emphasis filter coefficients. The filter used as emphasis must
        be a minimum phase filter. This means that all the poles and the zeroes are
        withing the unit circle. We can then invert the filter to apply de-emphasis.

        The filters.biquads.RBJ class can be used to generate a suitable emphasis
        filter.
        """
        assert repeats > 1, "at least two sequences are needed, (repeats=2)"

        _MLS_base.__init__(self, N=N, taps=taps)
        Audio.__init__(self, fs=fs, initialdata=self.get_full_sequence(repeats=repeats))

        self.repeats            = repeats
        self._length_impresp    = self.L/self.fs
        self._filter_emphasis   = Filter(B=B, A=A, fs=self.fs)
        self._filter_deemphasis = Filter(B=A, A=B, fs=self.fs) # inverse filter

        assert self._filter_emphasis.is_minimum_phase(), \
            "The emphasis filter must be minimum phase, i.e. possible to invert"

    def __repr__(self):
        B, A = self._filter_emphasis.get_coefficients()

        s = 'MLS(N=%i, taps=%s, fs=%r, repeats=%i, B=%s, A=%s)' \
            %(self.N, tuple(self.taps), self.fs, self.repeats, tuple(B), tuple(A))

        return s

    def __str__(self):
        B, A = self._filter_emphasis.get_coefficients()

        mls_string = _MLS_base.__str__(self)
        mls_string = "\n".join(mls_string.splitlines()[2:-1])

        s  = Audio.__str__(self)
        s += '%s\n'                             %mls_string
        s += 'repeats          : %i\n'          %self.repeats
        s += 'len(impulse)     : %.3f [s]\n'    %self._length_impresp
        s += 'emphasis filt. B : %s\n'          %str(B)
        s += 'emphasis filt. A : %s\n'          %str(A)
        return s

    def apply_emphasis(self):
        """Apply emphasis by filtering the whole audio signal. If high freqencies are
        boosted noise will be suppressed. If low frequencies are boosted we will get
        better signal to noise ratio in the lower area of the frequency response.
        """
        self._logger.debug("Applying emphasis filter, in place")
        self.samples = self._filter_emphasis.filter_samples(self.samples)

    def apply_deemphasis(self, x):
        """Undo the emphasis filter by filtering the signal with the inverse
        of the emphasis filter.
        """
        self._logger.debug("Applying de-emphasis filter")
        deemphasis_x = self._filter_deemphasis.filter_samples(x)
        return deemphasis_x

    def get_impulse(self, x):
        """Extract the impulse response. Returns an Audio instance.
        """
        imp = _MLS_base.get_impulse(self, x)
        y   = Audio(fs=self.fs, initialdata=imp)
        return y

class MLS_simple(object):
    def __init__(self, N=16, fs=96000, repeats=3):
        """Simplified usage for quick access to an MLS that performs the required
        message calls in the right order. Can also be used as an example on how
        to use the MLS class.

        Example:
            >>> mls = MLS_simple(N=18, fs=96000, repeats=4)
            >>> y = some_system_to_identify(mls.samples)
            >>> imp = mls.get_impulse(y)
            >>> imp.plot()
            >>> mls.plot_fft()

        The impulse response is stored as a member variable in this class after it
        has been extracted.
        """
        emphasis_filter = RBJ(filtertype="highshelf", gaindb=-10, f0=100, Q=0.707, fs=fs)
        B, A = emphasis_filter.get_coefficients()
        self._mls = MLS(N=N, taps=TAPS[N][0], fs=fs, repeats=repeats, B=B, A=A)
        self._mls.apply_emphasis()

        # map the name to the internal representation of the samples
        self.samples = self._mls.samples

    def __repr__(self):
        s = 'MLS_simple(N=%i fs=%r, repeats=%i)' %(self._mls.N, self._mls.fs, self._mls.repeats)
        return s

    def __str__(self):
        mls_string = str(self._mls)
        mls_string = "\n".join(mls_string.splitlines()[2:])

        s  = '=======================================\n'
        s += 'classname        : %s\n'      %self.__class__.__name__
        s += '%s' %str(mls_string)
        return s

    def get_impulse(self, x):
        """Extract the impulse response"""
        tmp = self._mls.apply_deemphasis(x)
        self._impulseresponse = self._mls.get_impulse(tmp)
        return self._impulseresponse

    def plot_fft(self, plotname=None):
        """Plot the magnitude response. Phase is not included in this plot."""
        assert hasattr(self, "_impulseresponse"), "call get_impulse(...) before trying to plot"

        # A window function on the impulse response does not work here, since
        # that would scale the impulse wrong.
        self._impulseresponse.plot_fft(plotname=plotname, window='rectangular', normalise=False)

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')
    fs  = 48000*2
    N   = 13

    #mls = _MLS_base(N=N, taps=TAPS[N][0])
    #mls = MLS(N=N, taps=TAPS[N][0], fs=fs)

    f = RBJ(filtertype="highshelf", gaindb=-10, f0=100, Q=0.707, fs=fs)
    B, A = f.get_coefficients()
    #mls = MLS(N=N, taps=TAPS[N][0], fs=fs, repeats=5, B=B, A=A)

    mls = MLS_simple(N=N, fs=fs, repeats=4)

    print (repr(mls))
    print(mls)

    y = mls.get_impulse(mls.samples)
    #mls.plot_fft()                                      # used with MLS_simple
    y.plot_fft(window='rectangular', normalise=False)   # used with MLS

    print('++ End of script ++')
