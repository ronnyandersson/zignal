'''
Created on Dec 31, 2013

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2013 Ronny Andersson
@license: MIT
'''

# standard library
import logging
import os
import types

# external libraries
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt

#===================================================================================================
# Classes
#===================================================================================================

class Audio(object):
    def __init__(self, channels=0, fs=96000, nofsamples=0, duration=None,
                 initialdata=None, dtype=np.float64):
        """Base class for audio processing. Samples are stored as a numpy array.

        We can create an instance by specifying a channel count and one of either
        a duration or a sample count parameter. The other way of creating an
        instance is by providing an already existing numpy array containing the
        audio samples.

        The shape of the audio samples are always (Nsamples_per_channel, Nchannels).
        """

        self._logger = logging.getLogger(__name__)

        # We sometimes divide by the sample rate to get time values
        assert fs > 0,  "sample rate cannot be zero or negative"

        self.fs         = fs    # sample rate should always be specified in the constructor
        self.nofsamples = None  # number of samples per channel
        self.duration   = None  # duration (length) in seconds
        self.ch         = None  # number of channels
        self._comment   = ''

        if initialdata is None:
            # if we are not given any initial samples we create an empty array of
            # zeros for the audio samples.
            assert isinstance(channels, int)
            assert not(nofsamples!=0 and duration is not None), "choose either samples or duration"

            self.ch = channels

            if duration is not None:
                self.nofsamples = int(duration*self.fs)
                self.duration   = duration
            else:
                self.nofsamples = nofsamples
                self._set_duration()

            # create space for the samples
            self.samples = np.zeros((self.nofsamples, self.ch), dtype=dtype)

        else:
            # An array of initial samples are given, use this to extract
            # channel count and durations.
            assert isinstance(initialdata, np.ndarray), \
                'Only numpy arrays are allowed as initial data'

            assert channels == 0, "parameter 'channels' is redundant if initial data is specified"
            assert nofsamples == 0, "parameter 'nofsamples' is redundant if initial data is specified"
            assert duration is None, "parameter 'duration' is redundant if initial data is specified"

            # copy the data to avoid unexpected data corruption
            self.samples = initialdata.copy()

            if self.samples.ndim == 1:
                # if the array is
                #     array([ 1.,  1.,  1.])
                # we expand it to
                #     array([[ 1.],
                #            [ 1.],
                #            [ 1.]])
                #
                self.samples = np.expand_dims(self.samples, axis=1)

            assert self.samples.ndim == 2, 'shape must be (Nsamples, Nchannels)'

            self.nofsamples, self.ch = self.samples.shape

            # initial data is assumed to have more samples than channels
            assert self.nofsamples > self.ch, 'shape must be (Nsamples, Nchannels)'

            self._set_duration()

        assert self.nofsamples is not None
        assert self.duration   is not None
        assert self.ch         is not None

    def __str__(self):
        s  = '=======================================\n'
        s += 'classname        : %s\n'          %self.__class__.__name__
        s += 'sample rate      : %.1f [Hz]\n'   %self.fs
        s += 'channels         : %i\n'          %self.ch
        s += 'duration         : %.3f [s]\n'    %self.duration
        s += 'datatype         : %s\n'          %self.samples.dtype
        s += 'samples per ch   : %i\n'          %self.nofsamples
        s += 'data size        : %.3f [Mb]\n'   %(self.samples.nbytes/(1024*1024))
        s += 'has comment      : %s\n'          %('yes' if len(self._comment)!=0 else 'no')
        if self.ch != 0:
            # += '-----------------:---------------------\n'
            s += 'peak             : %s\n'  %np.array_str(self.peak()[0],
                                                          precision=4, suppress_small=True)
            s += 'RMS              : %s\n'  %np.array_str(self.rms(),
                                                          precision=4, suppress_small=True)
            s += 'crestfactor      : %s\n'  %np.array_str(self.crest_factor(),
                                                          precision=4, suppress_small=True)
        s += '-----------------:---------------------\n'
        return s

    def __len__(self):
        return self.nofsamples

    def _set_duration(self):
        """internal method

        If we have modified the samples variable (by padding with zeros
        for example) we need to re-calculate the duration
        """
        self.duration = self.nofsamples/self.fs

    def _set_samples(self, idx=0, samples=None):
        """internal method

        NOTE: idx != channel

        idx is always zero indexed since it refers to the numpy array. Channels
        are always indexed from one since this is the natural way of identifying
        channel numbers.
        """
        assert isinstance(samples, np.ndarray)
        assert len(samples) == self.nofsamples
        self.samples[:,idx] = samples

    def pretty_string_samples(self, idx_start=0, idx_end=20, precision=4, header=False):
        s = ''
        if header:
            t = '  '
            u = 'ch'
            for i in range(self.ch):
                t += '-------:'
                u += '  %2i   :' %(i+1)
            t += '\n'
            u += '\n'

            s += t  #   -------:-------:-------:
            s += u  # ch   1   :   2   :   3   :
            s += t  #   -------:-------:-------:

        s += np.array_str(self.samples[idx_start:idx_end,:],
                          max_line_width=260,   # we can print 32 channels before linewrap
                          precision=precision,
                          suppress_small=True)
        if (idx_end-idx_start) < self.nofsamples:
            s  = s[:-1] # strip the right ']' character
            s += '\n ...,\n'
            lastlines = np.array_str(self.samples[-3:,:],
                                     max_line_width=260,
                                     precision=precision,
                                     suppress_small=True)
            s += ' %s\n' %lastlines[1:] # strip first '['
        return s

    def pad(self, nofsamples=0):
        """Zero pad *at the end* of the current audio data.

        increases duration by samples/fs
        """
        assert nofsamples >= 0, "Can't append negative number of samples"
        zeros = np.zeros((nofsamples, self.ch), dtype=self.samples.dtype)
        self.samples = np.append(self.samples, zeros, axis=0)

        self.nofsamples=len(self.samples)
        self._set_duration()

    def trim(self, start=None, end=None):
        """Trim samples **IN PLACE** """
        self.samples = self.samples[start:end]
        self.nofsamples=len(self.samples)
        self._set_duration()

    def _fade(self, millisec, direction):
        """Internal method.

        Fade in/out is essentially the same exept the slope (and position) of the
        ramp. Currently only a linear ramp is implemented.
        """
        assert np.issubdtype(self.samples.dtype, np.floating), \
            "only floating point processing implemented"
        assert millisec >= 0, "Got a time machine?"
        assert direction in ("in", "out")

        fade_seconds = millisec/1000
        assert self.duration > fade_seconds, "fade cannot be longer than the length of the audio"

        sample_count = np.ceil(fade_seconds*self.fs)
        self._logger.debug("fade %s sample count: %i" %(direction, sample_count))

        # generate the ramp
        if direction is "out":
            # ramp down
            ramp = np.linspace(1, 0, num=sample_count, endpoint=True)
        else:
            # ramp up
            ramp = np.linspace(0, 1, num=sample_count, endpoint=True)

        ones = np.ones(len(self)-len(ramp))

        # glue the ones and the ramp together
        if direction is "out":
            gains = np.append(ones, ramp, axis=0)
        else:
            gains = np.append(ramp, ones, axis=0)

        # expand the dimension so we get a one channels array of samples,
        # as in (samples, channels)
        gains = np.expand_dims(gains, axis=1)

        assert len(gains) ==  len(self)

        # repeat the gain vector so we get as many gain channels as all the channels
        gains = np.repeat(gains, self.ch, axis=1)

        assert gains.shape == self.samples.shape

        # apply gains
        self.samples = self.samples * gains

    def fade_in(self, millisec=10):
        """Fade in over 'millisec' seconds. Applies on *all* channels"""
        self._fade(millisec, "in")

    def fade_out(self, millisec=30):
        """Fade out over 'millisec' seconds. Applies on *all* channels"""
        self._fade(millisec, "out")

    def delay(self, n, channel=1):
        """Delay channel x by n samples"""
        self.samples[:,channel-1] = np.pad(self.samples[:,channel-1], (n, 0),
                                           mode="constant")[:-n]

    def get_time(self):
        """Return a vector of time values, starting with t0=0. Useful when plotting."""
        return np.linspace(0, self.duration, num=self.nofsamples, endpoint=False)

    def comment(self, comment=None):
        """Modify or return a string comment."""
        assert isinstance(comment, (str, type(None))), "A comment is a string"

        if comment is not None:
            self._comment = comment

        return self._comment

    def append(self, *args):
        """Add (append) channels *to the right* of the current audio data.

        does zeropadding
        increases channel count
        """
        for i, other in enumerate(args):
            assert isinstance(other, Audio), "only Audio() instances can be used"

            self._logger.debug("** iteration %02i --> appending %s" %((i+1), other.__class__.__name__))
            assert self.fs == other.fs, "Sample rates must match (%s != %s)" %(self.fs, other.fs)
            assert self.samples.dtype == other.samples.dtype, \
                "Data types must match (%s != %s)"%(self.samples.dtype, other.samples.dtype)

            max_nofsamples = max(self.nofsamples,  other.nofsamples)
            missingsamples = abs(self.nofsamples - other.nofsamples)

            self._logger.debug("max nof samples: %i" %max_nofsamples)
            self._logger.debug("appending %i new channel(s) and %i samples" %(other.ch, missingsamples))

            if self.nofsamples > other.nofsamples:
                self._logger.debug("self.nofsamples > other.nofsamples")

                tmp = np.append(other.samples,
                                np.zeros(((missingsamples), other.ch), dtype=other.samples.dtype),
                                axis=0)
                self.samples = np.append(self.samples, tmp, axis=1)

            elif self.nofsamples < other.nofsamples:
                self._logger.debug("self.nofsamples < other.nofsamples")

                tmp = np.append(self.samples,
                                np.zeros(((missingsamples), self.ch), dtype=self.samples.dtype),
                                axis=0)
                self.samples = np.append(tmp, other.samples, axis=1)

            else:
                self._logger.debug("self.nofsamples == other.nofsamples")
                self.samples = np.append(self.samples, other.samples, axis=1)

            self.ch = self.ch+other.ch
            self.nofsamples=max_nofsamples
            self._set_duration()

    def concat(self, *args):
        """Concatenate (append) samples *after* the current audio data.

        example:
            x1 = 1234
            x2 = 5678
            x1.concat(x2) --> 12345678

        """
        for i, other in enumerate(args):
            assert isinstance(other, Audio), "only Audio() instances can be used"

            self._logger.debug("** iteration %02i --> appending %s" %((i+1), other.__class__.__name__))
            assert self.fs == other.fs, "Sample rates must match (%s != %s)" %(self.fs, other.fs)
            assert self.samples.dtype == other.samples.dtype, \
                "Data types must match (%s != %s)"%(self.samples.dtype, other.samples.dtype)
            assert self.ch == other.ch, "channel count must match"

            self.samples = np.append(self.samples, other.samples, axis=0)

            self.nofsamples=len(self.samples)
            self._set_duration()

    def gain(self, *args):
        """Apply gain to the audio samples. Always specify gain values in dB.

        Converts **IN PLACE**
        """
        self._logger.debug('gains: %s' %str(args))

        dt  = self.samples.dtype
        lin = db2lin(args)

        # apply the (linear) gain
        self.samples = lin*self.samples

        # make sure that the data type is retained
        self.samples = self.samples.astype(dt)

    def rms(self):
        """Calculate the RMS (Root Mean Square) value of the audio
        data. Returns the RMS value for each individual channel
        """
        if not (self.samples == 0).all():
            if np.issubdtype(self.samples.dtype, np.floating):
                rms = np.sqrt(np.mean(np.power(self.samples, 2), axis=0))
            else:
                # use a bigger datatype for ints since we most likely will
                # overflow when calculating to the power of 2
                bigger  = np.asarray(self.samples, dtype=np.int64)
                rms     = np.sqrt(np.mean(np.power(bigger, 2), axis=0))

        elif len(self.samples) == 0:
            # no samples are set but channels are configured
            rms = np.zeros(self.ch)
            rms[:] = float('nan')
        else:
            rms = np.zeros(self.ch)
        return rms

    def peak(self):
        """Calculate peak sample value (with sign)"""

        if len(self.samples) != 0:
            if np.issubdtype(self.samples.dtype, np.floating):
                idx = np.absolute(self.samples).argmax(axis=0)
            else:
                # We have to be careful when checking two's complement since the absolute value
                # of the smallest possible value can't be represented without overflowing. For
                # example: signed 16bit has range [-32768, 32767] so abs(-32768) cannot be
                # represented in signed 16 bits --> use a bigger datatype
                bigger  = np.asarray(self.samples, dtype=np.int64)
                idx     = np.absolute(bigger).argmax(axis=0)

            peak = np.array([self.samples[row,col] for col, row in enumerate(idx)])
        else:
            # no samples are set but channels are configured
            idx  = np.zeros(self.ch, dtype=np.int64)
            peak = np.zeros(self.ch)
            peak[:] = float('nan')

        return peak, idx

    def crest_factor(self):
        """Calculate the Crest Factor (peak over RMS) value of the
        audio. Returns the crest factor value for each channel.
        Some common crest factor values:

            sine     : 1.414...
            MLS      : 1    (if no emphasis filter is applied)
            impulse  : very high. The value gets higher the longer
                       the length of the audio data.
            square   : 1    (ideal square)
            zeros    : NaN (we cannot calculate 0/0)

        """
        rms = self.rms()
        assert len(rms) != 0

        with np.errstate(invalid='ignore'):
            # if the rms is zero we will get division errors. Ignore them.
            if len(self.samples) != 0:
                crest = np.abs(self.samples).max(axis=0)/rms
            else:
                # no samples are set but channels are configured
                crest = np.zeros(self.ch)
                crest[:] = float('nan')

        return crest

    def convert_to_integer(self, targetbits=16):
        """Scale floating point values between [-1.0, 1.0] to the equivalent
        signed integer value. Converts **IN PLACE**

        Note: 24 bit signed integers and 8 bit unsigned integers currently unsupported.
        """
        assert targetbits in (8, 16, 32, 64)
        assert self.samples.dtype in (np.int8, np.int16, np.int32, np.int64,
                                      np.float32, np.float64)
        dt = { 8 : 'int8',
              16 : 'int16',
              32 : 'int32',
              64 : 'int64'}

        sourcebits = self.samples.itemsize * 8

        if self.samples.dtype in (np.float32, np.float64):
            self._logger.debug("source is %02i bits (float),   target is %2i bits (integer)"
                              %(sourcebits, targetbits))

            self.samples = np.array(self.samples*(2**(targetbits-1)-1),
                                    dtype=dt.get(targetbits))
        else:
            self._logger.debug("source is %02i bits (integer), target is %2i bits (integer)"
                              %(sourcebits, targetbits))
            raise NotImplementedError("TODO: implement scale int->int")

    def convert_to_float(self, targetbits=64):
        """Scale integer values to equivalent floating point values
        between [-1.0, 1.0]. Converts **IN PLACE**
        """
        assert targetbits in (32, 64)
        assert self.samples.dtype in (np.int8, np.int16, np.int32, np.int64,
                                      np.float32, np.float64)
        dt = {32 : 'float32',
              64 : 'float64'}

        sourcebits = self.samples.itemsize * 8

        if self.samples.dtype in (np.int8, np.int16, np.int32, np.int64):
            self._logger.debug("source is %02i bits (integer), target is %2i bits (float)"
                              %(sourcebits, targetbits))

            self.samples = np.array(self.samples/(2**(sourcebits-1)), dtype=dt.get(targetbits))

        else:
            self._logger.debug("source is %02i bits (float),   target is %2i bits (float)"
                              %(sourcebits, targetbits))

            self.samples = np.array(self.samples, dtype=dt.get(targetbits))

    def write_wav_file(self, filename=None):
        """Save audio data to .wav file."""

        assert filename is not None, "Specify a filename, for example 'filename=audio.wav'"

        self._logger.debug("writing file %s" %filename)
        if self.samples.dtype == np.float64:
            self._logger.warn("datatype is %s" %self.samples.dtype)

        try:
            scipy.io.wavfile.write(filename, self.fs, self.samples)
        except:
            self._logger.exception("Could not write file: '%s'" %filename)

    def plot(self, ch=1, plotname=None, **kwargs):
        """Plot the audio data on a time domain plot.

        example:

            x1 = Sinetone(f0=0.2, fs=10, nofsamples=50)
            x1.plot(linestyle='--', marker='x', color='r', label='sine at 0.2Hz')

        """
        # TODO: add range to plotdata [None:None] is everything

        if ch != 'all':
            assert ch-1 < self.ch, "channel does not exist"

        plt.figure(1)
        plt.title("%s" %self.__class__.__name__)
        if ch != 'all':
            plt.plot(self.get_time(), self.samples[:,ch-1], **kwargs)
        else:
            plt.plot(self.get_time(), self.samples, **kwargs)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude [linear]')
        if 'label' in kwargs:
            plt.legend(loc='best')
        plt.grid(True)

        if plotname is None:
            plt.show()
        else:
            plt.savefig(plotname)
            plt.close(1)

    def plot_fft(self, plotname=None, window='hann', normalise=True, **kwargs):
        """Make a plot (in the frequency domain) of all channels"""

        ymin = kwargs.get('ymin', -160) #dB

        freq, mag = self.fft(window=window, normalise=normalise)

        fig_id = 1
        plt.figure(fig_id)

        #plt.semilogx(freq, mag, **kwargs)   # plots all channel directly
        for ch in range(self.ch):
            plt.semilogx(freq, mag[:,ch], label='ch%2i' %(ch+1))

        plt.xlim(left=1)    # we're not interested in freqs. below 1 Hz
        plt.ylim(bottom=ymin)

        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude [dB]')

        plt.legend(loc='best')
        plt.grid(True)

        if plotname is None:
            plt.show()
        else:
            plt.savefig(plotname)
            plt.close(fig_id)

    def fft(self, window='hann', normalise=True):
        """Calculate the FFT of all channels. Returns data up to fs/2"""
        fftsize = self.nofsamples

        # Avoid Mersenne Primes
        if fftsize in [(2**13)-1, (2**17)-1, (2**19)-1, (2**31)-1]:
            self._logger.warn("FFT size is a Mersenne Prime, increasing size by 1")
            fftsize = fftsize+1

        self._logger.debug("fftsize: %i" %fftsize)
        self._logger.debug("window : %s" %str(window))

        win = scipy.signal.windows.get_window(window, Nx=self.nofsamples) # not fftsize!
        win = np.expand_dims(win, axis=1)
        y   = self.samples*win

        Y   = np.fft.fft(y, n=fftsize, axis=0)
        if normalise:
            Y = Y/fftsize

        mag = lin2db(np.abs(Y))
        frq = np.fft.fftfreq(fftsize, 1/self.fs)

        frq = frq[:int(fftsize/2)]
        mag = mag[:int(fftsize/2)]

        return frq, mag

    def dither(self, bits=16, distribution='TPDF'):
        raise NotImplementedError('TODO')

        assert distribution == 'TPDF', \
            "Only the Triangular Probability Density Function is implemented"

        # Triangular Probability Density Function
        noise = np.random.triangular(-1, 0, 1, self.samples.shape)

    def resample(self, targetrate=44100):
        raise NotImplementedError('TODO')

    def set_sample_rate(self, new_fs):
        """Change the sample rate fs *without* up/down sample conversion.
        This would be the same as changing the playback speed. All data is
        left intact and only the time parameters (fs and duration) are
        changed.
        """
        ratio   = new_fs/self.fs
        self.fs = new_fs

        self._logger.debug('ratio: %.3f' %ratio)
        self._set_duration()
        return ratio

    def normalise(self):
        """Normalise samples so that the new range is
        [-1.0,  1.0] for floats

        Converts **IN PLACE**

        TODO: verify
        [-2^n, 2^n-1] for ints
        """
        peaks, unused_idx = self.peak()
        self._logger.debug("raw peaks: %s" %peaks)

        max_abs = np.max(np.absolute(peaks))
        self._logger.debug("max_abs: %s" %max_abs)

        self.samples = self.samples/max_abs

        peaks, unused_idx = self.peak()
        self._logger.debug("new peaks: %s" %peaks)

#===================================================================================================
# Audio sub-classes
#===================================================================================================

class Sinetone(Audio):
    def __init__(self, f0=997, fs=96000, duration=None, gaindb=0, nofsamples=0, phasedeg=0):
        """Generate a sine tone"""
        assert f0 < fs/2, "Sampling theorem is violated"
        Audio.__init__(self, channels=1, fs=fs, nofsamples=nofsamples, duration=duration)

        self.f0         = f0
        self.phasedeg   = phasedeg

        self._set_samples(idx=0, samples=self._sine_gen(f0, phasedeg))
        self.gain(gaindb)

    def _sine_gen(self, freq, pha):
        return np.sin(2*np.pi*freq*self.get_time()+np.deg2rad(pha))

    def __repr__(self):
        assert self.ch == 1, "If a channel has been appended we don't know anything about its data"

        s = 'Sinetone(f0=%r, fs=%r, nofsamples=%r, gaindb=%r, phasedeg=%r)' \
            %(self.f0, self.fs, self.nofsamples,
              lin2db(abs(float(self.peak()[0]))),  # only one channel here.
              self.phasedeg)
        return s

    def __str__(self):
        s  = Audio.__str__(self)
        s += 'frequency        : %.1f [Hz]\n'   %self.f0
        s += 'phase            : %.1f [deg]\n'  %self.phasedeg
        s += '-----------------:---------------------\n'
        return s

    def set_sample_rate(self, new_fs):
        ratio = Audio.set_sample_rate(self, new_fs)
        self.f0 = ratio * self.f0

class Sinetones(Sinetone):
    def __init__(self, *args, **kwargs):
        """Generate multiple sinetones. This is a quick way to generate multichannel audio.
        Each positional argument generates a sinetone at that channel. Setting the frequency
        to 0 guarantees that the channel is muted (contains samples with the value 0).
        Keywords accepted are similar to the ones used in the Sinetone() class.

        Example:

            >>> x = Sinetones(200, 500, 900, fs=24000, duration=1.5, gaindb=-6, phasedeg=0)
            >>> print(x)
            =======================================
            classname        : Sinetones
            sample rate      : 24000.0 [Hz]
            channels         : 3
            duration         : 1.500 [s]
            datatype         : float64
            samples per ch   : 36000
            data size        : 0.824 [Mb]
            has comment      : no
            peak             : [ 0.5012  0.5012 -0.5012]
            RMS              : [ 0.3544  0.3544  0.3544]
            crestfactor      : [ 1.4142  1.4142  1.4142]
            -----------------:---------------------
            phase (all ch)   : 0.0 [deg]
                             :
            channel  1       : 200.0 [Hz]
            channel  2       : 500.0 [Hz]
            channel  3       : 900.0 [Hz]
            -----------------:---------------------
            >>>

        The gaindb argument can be an iterable of the same length as the number of frequencies
        specified. In this case a gain can be applied individually for each channel.

            >>> x = Sinetones(1000, 2000, duration=1, gaindb=(-6, -20))

        A list can be used as the argument for the frequencies. Use the * notation to expand
        the list:

            >>> import numpy as np
            >>> f = np.zeros(8)
            >>> f[3] = 700
            >>> f[7] = 2000
            >>> x = Sinetones(*f, duration=1)
            >>> print(x)
            =======================================
            classname        : Sinetones
            sample rate      : 96000.0 [Hz]
            channels         : 8
            duration         : 1.000 [s]
            datatype         : float64
            samples per ch   : 96000
            data size        : 5.859 [Mb]
            has comment      : no
            peak             : [ 0.  0.  0. -1.  0.  0.  0.  1.]
            RMS              : [ 0.      0.      0.      0.7071  0.      0.      0.      0.7071]
            crestfactor      : [    nan     nan     nan  1.4142     nan     nan     nan  1.4142]
            -----------------:---------------------
            phase (all ch)   : 0.0 [deg]
                             :
            channel  1       :
            channel  2       :
            channel  3       :
            channel  4       : 700.0 [Hz]
            channel  5       :
            channel  6       :
            channel  7       :
            channel  8       : 2000.0 [Hz]
            -----------------:---------------------
            >>>

        The argument phasedeg applies to all channels.
        """

        fs                  = kwargs.pop('fs',          96000)
        duration            = kwargs.pop('duration',    None)
        nofsamples          = kwargs.pop('nofsamples',  0)
        self._gaindb        = kwargs.pop('gaindb',      0)
        self.phasedeg       = kwargs.pop('phasedeg',    0)
        self.frequencies    = args

        for frequency in self.frequencies:
            assert frequency < fs/2, "Sampling theorem is violated for frequency %.1f" %frequency

        if not isinstance(self._gaindb, int):
            assert len(self._gaindb) == len(self.frequencies), \
                "set as many gains as channels used: %i != %i" %(len(self._gaindb),
                                                                 len(self.frequencies))

        Audio.__init__(self, channels=len(self.frequencies), fs=fs, nofsamples=nofsamples,
                       duration=duration)

        for i, frequency in enumerate(self.frequencies):
            if frequency != 0:
                self._set_samples(idx=i, samples=self._sine_gen(frequency, self.phasedeg))
            else:
                pass # channel is silence

        self.gain(self._gaindb)

    def __repr__(self):
        s = 'Sinetones(*%r, fs=%r, nofsamples=%r, gaindb=%r, phasedeg=%r)' \
            %(list(self.frequencies), self.fs, self.nofsamples,self._gaindb,self.phasedeg)
        return s

    def __str__(self):
        s  = Audio.__str__(self)
        s += 'phase (all ch)   : %.1f [deg]\n'  %self.phasedeg
        s += '                 :\n'
        for i, frequency in enumerate(self.frequencies):
            if frequency != 0:
                s += 'channel %2i       : %.1f [Hz]\n'   %(i+1, frequency)
            else:
                s += 'channel %2i       :\n'   %(i+1)
        s += '-----------------:---------------------\n'
        return s

    def set_sample_rate(self, new_fs):
        ratio = Audio.set_sample_rate(self, new_fs)
        self.frequencies = [ratio*f for f in self.frequencies]

class SquareWave(Audio):
    def __init__(self, f0=997, fs=96000, duration=None, gaindb=0, nofsamples=0,
                 phasedeg=0, dutycycle=0.5):
        """Generate an ideal squarewave."""
        assert f0 < fs/2, "Sampling theorem is violated"
        assert dutycycle < 1 and dutycycle > 0
        Audio.__init__(self, channels=1, fs=fs, nofsamples=nofsamples, duration=duration)

        self.f0         = f0
        self.phasedeg   = phasedeg
        self.dutycycle  = dutycycle

        samples = scipy.signal.square(2*np.pi*f0*self.get_time()+np.deg2rad(phasedeg),
                                      duty=dutycycle)
        self._set_samples(idx=0, samples=samples)
        self.gain(gaindb)

    def __repr__(self):
        assert self.ch == 1, "If a channel has been appended we don't know anything about its data"

        s = 'SquareWave(f0=%r, fs=%r, gaindb=%r, nofsamples=%r, phasedeg=%r, dutycycle=%r)' \
            %(self.f0, self.fs,
              lin2db(abs(float(self.peak()[0]))), # only one channel here.
              self.nofsamples, self.phasedeg, self.dutycycle)
        return s

    def __str__(self):
        s  = Audio.__str__(self)
        s += 'frequency        : %.1f [Hz]\n'       %self.f0
        s += 'phase            : %.1f [deg]\n'      %self.phasedeg
        s += 'duty cycle       : %.3f (%4.1f%%)\n'  %(self.dutycycle, self.dutycycle*100)
        s += '-----------------:---------------------\n'
        return s

    def set_sample_rate(self, new_fs):
        ratio = Audio.set_sample_rate(self, new_fs)
        self.f0 = ratio * self.f0

class FourierSeries(Sinetone):
    def __init__(self, f0=997, fs=96000, duration=None, gaindb=0, nofsamples=0,
                 phasedeg=0, harmonics=7,):
        """Construct a square wave by adding odd harmonics with decreasing
        amplitude, i.e. Fourier Series.
        """
        Sinetone.__init__(self, f0=f0, phasedeg=phasedeg, fs=fs, nofsamples=nofsamples,
                          duration=duration, gaindb=0)

        assert harmonics >= 0

        self.harmonics = harmonics
        self._logger.debug("fundamental f0: %.1f" %f0)

        for n in range(3, 2*(self.harmonics+1), 2):
            if n <= 15:
                self._logger.debug("adding harmonic n: %2i with amplitude 1/%i" %(n, n))
            if n == 17:
                self._logger.debug("adding %i more harmonics..." %(self.harmonics-(n-3)//2))

            #self.samples[:,0] += np.sin(2*np.pi*(n*f0)*self.get_time()+np.deg2rad(phasedeg*n))/n
            self.samples[:,0] += (1/n)*self._sine_gen(n*f0, n*phasedeg)
        self.gain(gaindb)

    def __repr__(self):
        assert self.ch == 1, "If a channel has been appended we don't know anything about its data"

        s = 'FourierSeries(f0=%r, fs=%r, gaindb=%r, nofsamples=%r, phasedeg=%r, harmonics=%r)' \
            %(self.f0, self.fs,
              lin2db(abs(float(self.peak()[0]))), # only one channel here.
              self.nofsamples, self.phasedeg, self.harmonics)
        return s

    def __str__(self):
        s  = Sinetone.__str__(self)
        s = s.rstrip('-----------------:---------------------\n')
        s += '\n'
        s += 'harmonics        : %i \n' %self.harmonics
        s += '-----------------:---------------------\n'
        return s

class Noise(Audio):
    colours = ('white', 'pink', 'brown', 'blue', 'violet', 'grey')

    def __init__(self, channels=1, fs=96000, duration=None, gaindb=-10, nofsamples=0,
                 colour='white'):
        """Generate uncorrelated noise.

        white       : flat power spectral density
        pink        : -3dB per octave
        brown(ian)  : -6dB per octave
        blue        : +3dB per octave
        violet      : +6dB per octave
        grey        : equal loudness
        """

        assert colour in Noise.colours, "choose the colour of the noise: %s" %str(Noise.colours)
        Audio.__init__(self, channels=channels, fs=fs, nofsamples=nofsamples, duration=duration)
        # the distribution in np.random.uniform is half open, i.e -1.0 is
        # included but 1.0 is not. Possible fix: use integers instead, then
        # scale to floats. Might not work, since the integers will be
        # represented using twos complement and we then have an asymmetrical
        # range anyhow.

        self._colour = colour

        # first generate uniformly distributed noise, i.e. white noise. Then filter
        # to get the required shape.
        for ch in range(channels):
            self._set_samples(idx=ch,
                              samples=np.random.uniform(low=-1.0, high=1.0, size=self.nofsamples))
        if self._colour=='pink':
            # -3dB per octave
            self._logger.debug("filtering to get pink noise")
            # http://dsp.stackexchange.com/q/322/6999
            B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            A = [1, -2.494956002, 2.017265875, -0.522189400]
            self.samples = scipy.signal.lfilter(B, A, self.samples, axis=0)

        elif self._colour=='brown':
            # -6dB per octave
            raise NotImplementedError('TODO')

        elif self._colour=='blue':
            # +3dB per octave
            raise NotImplementedError('TODO')

        elif self._colour=='violet':
            # +6dB per octave
            raise NotImplementedError('TODO')

        elif self._colour=='grey':
            # equal loudness
            raise NotImplementedError('TODO')

        self.gain(gaindb)

    def __str__(self):
        s  = Audio.__str__(self)
        s += 'colour           : %s\n'  %self._colour
        s += '-----------------:---------------------\n'
        return s

class WavFile(Audio):
    def __init__(self, filename=None, scale2float=True):
        """Read a .wav file from disk"""
        assert filename is not None, "Specify a filename"
        self.filename = filename

        fs, samples = scipy.io.wavfile.read(filename)
        if samples.ndim == 1:
            samples = np.expand_dims(samples, axis=1)

        Audio.__init__(self, fs=fs, initialdata=samples)

        del samples # just to make sure

        if scale2float:
            self.convert_to_float(targetbits=64)

    def __str__(self):
        s  = Audio.__str__(self)
        s += 'filename         : %s\n'  %os.path.basename(self.filename)
        s += '-----------------:---------------------\n'
        return s

#===================================================================================================
# Functions
#===================================================================================================

def lin2db(lin):
    with np.errstate(divide='ignore'):
        # linear value 0 is common (as -inf dB) so we ignore any division warnings
        db = 20*np.log10(lin)
    return db

def pow2db(power):
    with np.errstate(divide='ignore'):
        # ignore any division warnings
        db = 10*np.log10(power)
    return db

def db2lin(db):
    lin = np.power(10, np.array(db)/20)
    return lin

def db2pow(db):
    power = np.power(10, np.array(db)/10)
    return power

def speed_of_sound(temperature=20, medium='air'):
    """The speed of sound is depending on the medium and the temperature. For air at
    a temperature of 20 degree Celcius the speed of sound is approximately 343 [m/s]
    """
    assert medium in ['air',], "TODO: water, iron"

    c = float('nan')

    if medium == 'air':
        c = 331.3*np.sqrt(1+temperature/273.15)

    return c

def wavelength(frequency, speed=343.2):
    """Calculate the wavelength l of frequency f given the speed (of sound)"""
    l = speed/frequency
    return l

def rad2hz(w0, fs=96000):
    """Calculate a normalised rotational frequency so that w0=2*pi --> f0=fs

                    w0
        f0 = fs * ------
                   2*pi
    """
    return fs*np.array(w0)/(2*np.pi)

def hz2rad(f0, fs=96000):
    """Calculate a normalised angular frequency so that f0=fs --> w0=2*pi

               1
        w0 = ----- * 2*pi*f0
              fs
    """
    return (1/fs)*2*np.pi*np.array(f0)

__all__ = [
           # classes
           'Audio',
           'Sinetone',
           'Sinetones',
           'SquareWave',
           'FourierSeries',
           'Noise',
           'WavFile',

           # functions
           'lin2db',
           'pow2db',
           'db2lin',
           'db2pow',
           'speed_of_sound',
           'wavelength',
           'rad2hz',
           'hz2rad',
           ]

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    print('-- Done --')
