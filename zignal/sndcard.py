"""
Created on 15 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
"""

# standard library
from abc import ABCMeta, abstractmethod
import logging
import warnings

# external libraries
import numpy as np
try:
    import pyaudio
except ImportError:
    warnings.warn("PyAudio not found. Will not be able to create sndcard instances", category=ImportWarning)

# local libraries
from zignal import Audio, Noise

def list_devices():
    """List all available sound cards."""
    return PA.list_devices()

#===================================================================================================
# Abstract Base Class, inherit and implement the methods marked as @abstractmethod
#===================================================================================================
class _Device(object, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self._logger = logging.getLogger(__name__)

    def __str__(self):
        s  = '=======================================\n'
        s += 'classname        : %s\n' %self.__class__.__name__
        return s

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        self._logger.debug("--- open")

    def close(self):
        self._logger.debug("--- close")

    @abstractmethod
    def play(self, x, **kwargs):
        """Play audio"""
        self._logger.debug("--- play")
        assert isinstance(x, Audio)

    @abstractmethod
    def rec(self, duration=None, channels=1, fs=96000, **kwargs):
        """Record audio"""
        self._logger.debug("--- rec")
        assert duration is not None, "Specify a duration (in seconds) to record for"

    @abstractmethod
    def play_rec(self, x, **kwargs):
        """Play and record audio"""
        self._logger.debug("--- play_rec")
        assert isinstance(x, Audio)

#===================================================================================================
# Stub class
#===================================================================================================
class Stub(_Device):
    """Stub device that can be dropped in anywhere as a fake sound card.
    The record methods will return noise. This is intended to be used
    during development when a real device would block.
    """

    def play(self, x, **kwargs):
        _Device.play(self, x, **kwargs)
        self._logger.warn("*** Stub play")

    def rec(self, duration=None, channels=1, fs=96000, **kwargs):
        _Device.rec(self, duration=duration, channels=channels, fs=fs, **kwargs)
        self._logger.warn("*** Stub rec")

        # fake a signal with white noise
        n = Noise(channels=channels, fs=fs, duration=duration, gaindb=-60)
        n.convert_to_float(targetbits=32)
        y = Audio(fs=fs, initialdata=n.samples)
        return y

    def play_rec(self, x, **kwargs):
        _Device.play_rec(self, x, **kwargs)
        self._logger.warn("*** Stub play_rec")

        # fake a signal with white noise
        n = Noise(channels=x.ch, fs=x.fs, nofsamples=x.nofsamples, gaindb=-60)
        n.convert_to_float(targetbits=32)
        y = Audio(fs=x.fs, initialdata=n.samples)
        return y

#===================================================================================================
# PyAudio (Portaudio) implementation
#===================================================================================================
class PA(_Device):
    """PyAudio wrapper. Uses the Audio base class as input signal and returns
    Audio instances after a recording. This implementation is using the blocking
    strategy."""

    #--------------------------------------------------------------------------------
    #    Portaudio frame
    #
    #    http://music.columbia.edu/pipermail/portaudio/2007-January/006615.html
    #
    #    A 'frame' is all data required at a snapshot in time, where snapshots
    #    are the same as samplerate. This means that 2 ch of 16bit audio is
    #    a frame of size 32 bits
    #
    #            +------------+------------+
    #    stereo  | 16bit ch1  | 16bit ch2  |
    #            +------------+------------+
    #
    #                         |
    #                         v
    #
    #            +-------------------------+
    #            |         4 bytes         |
    #            +-------------------------+
    #
    #    frame size calculation:
    #    2 bytes per sample, 2 channels --> 2*2=4 bytes
    #
    #    Another example:
    #    32 bit float, 8 channels --> 4 bytes per channel, 8 channels == 32 bytes
    #
    #            +-----+-----+-----+-----+-----+-----+-----+-----+
    #            | ch1 | ch2 | ch3 | ch4 | ch5 | ch6 | ch7 | ch8 |
    #            +-----+-----+-----+-----+-----+-----+-----+-----+
    #
    #            +-----------------------------------------------+
    #            |                   (32 bytes)                  |
    #            +-----------------------------------------------+
    #
    #    In other words, because we have audio data in a numpy vector
    #    where (rows,colums) --> (samples, channels) this means that each *row*
    #    is a frame.
    #
    #--------------------------------------------------------------------------------

    def __init__(self, device_out='default', device_in='default'):
        """Set the device_out and/or device_in string based on the name of the
        sound card. An int can also be used if the id is known beforehand. Note that
        the id can change when another sound card is detected, for example when a USB
        card is connected. The available sound cards can be found by calling list_devices()
        """
        _Device.__init__(self)

        self._device_out = device_out
        self._device_in  = device_in

        if isinstance(device_out, int):
            self._index_out = device_out
        else:
            self._index_out = self._get_id(name=device_out, find_output=True)

        if isinstance(device_in, int):
            self._index_in = device_in
        else:
            self._index_in = self._get_id(name=device_in,  find_output=False)

    def __str__(self):
        s  = _Device.__str__(self)
        s += 'portaudio        : %s %s\n'       %(pyaudio.get_portaudio_version(),
                                                  pyaudio.get_portaudio_version_text())
        s += 'pyaudio          : %s\n'          %pyaudio.__version__
        s += 'output device    : id %i, %s\n'   %(self._index_out, self._device_out)
        s += 'input device     : id %i, %s\n'   %(self._index_in,  self._device_in)
        return s

    def _get_id(self, name=None, find_output=True):
        """Find the id of the sound card that matches the string name"""
        retval = -1
        pa_get_id = pyaudio.PyAudio()

        try:
            if name=='default':
                if find_output:
                    device = pa_get_id.get_default_output_device_info()
                    if device['maxOutputChannels'] > 0:
                        self._device_out = device['name']
                        retval = device['index']

                else:
                    device = pa_get_id.get_default_input_device_info()
                    if device['maxInputChannels'] > 0:
                        self._device_in = device['name']
                        retval = device['index']
            else:
                for idx in range(pa_get_id.get_device_count()):
                    device = pa_get_id.get_device_info_by_index(idx)

                    if find_output:
                        if device['maxOutputChannels'] > 0:
                            if device['name'] == name:
                                retval = idx
                                break
                    else:
                        if device['maxInputChannels'] > 0:
                            if device['name'] == name:
                                retval = idx
                                break
        finally:
            pa_get_id.terminate()

        if retval == -1:
            s = "Device '%s' not found. Check the inputs and outputs arguments" %name
            print(s)
            try:
                print("Available devices: \n%s" %self.list_devices())
            finally:
                raise LookupError(s)

        return retval

    def open(self):
        """Open a PyAudio instance. This needs to be called before play(),
        play_rec() or rec() is called. This can be done in two ways:

            snd = PA()
            snd.open()
            try:
                snd.play(x)
            finally:
                snd.close()

        or use the 'with' statement:

            with PA() as snd:
                snd.play(x)

        """
        self._logger.debug("creating pyaudio instance")
        self.pa = pyaudio.PyAudio()

    def close(self):
        """Terminate the PyAudio instance. Must be called if open() has been called"""
        self.pa.terminate()
        self._logger.debug("pyaudio instance terminated")

    @classmethod
    def list_devices(cls):
        """Get a pretty string with all available sound cards.

        When using a portaudio instance, the id of the sound device needs
        to be known. This method is listing the available devices so that
        the id can be found.
        """

        s  = ''
        s += '--------------------------------------------------------------------\n'
        s += 'id out  in  def.fs   API            name\n'
        s += '--------------------------------------------------------------------\n'
        #--->| 0   2   2  44100.0  ALSA           Intel 82801AA-ICH: - (hw:0,0)
        pa_list_dev = pyaudio.PyAudio()
        try:
            for idx in range(pa_list_dev.get_device_count()):
                device = pa_list_dev.get_device_info_by_index(idx)
                s+='%2i %3i %3i %8.1f  %s %s\n' %(
                    device['index'],
                    device['maxOutputChannels'],
                    device['maxInputChannels'],
                    device['defaultSampleRate'],
                    pa_list_dev.get_host_api_info_by_index(device['hostApi'])['name'].ljust(len('Windows WASAPI')),
                    device['name'],
                    )
            s += '\n'
            s += 'default output device id: %i\n' %pa_list_dev.get_default_output_device_info()['index']
            s += 'default input  device id: %i\n' %pa_list_dev.get_default_input_device_info()['index']
            s += '--------------------------------------------------------------------\n'
        finally:
            pa_list_dev.terminate()

        return s

    def _data_format(self, x):
        """The data types in numpy needs to be mapped to the equivalent type in
        portaudio. This is an issue for 24 bit audio files since there isn't a
        24 bit data type in numpy. This is currently not implemented. There are
        some options on how to do this. We could for example use a 32 bit int and
        store the 24 bits either so that bits 1 to 8 is set to zeroes or so that
        bits 25 to 32 is set to zeros.
        """
        retval = None

        if x.samples.dtype == np.dtype(np.float32):
            self._logger.debug("pyaudio.paFloat32")
            retval = pyaudio.paFloat32
        elif x.samples.dtype == np.dtype(np.int16):
            self._logger.debug("pyaudio.paInt16")
            retval = pyaudio.paInt16
        elif x.samples.dtype == np.dtype(np.int32):
            self._logger.debug("pyaudio.paInt32")
            retval = pyaudio.paInt32
        else:
            raise NotImplementedError("Data type not understood: %s" %x.samples.dtype)

        return retval

    def _check_pow2(self, n):
        """Check that buffer size is a power of 2 (32, 64, ..., 1024, 2048, ...)"""
        check = 2**int(np.round(np.log2(n))) == n
        return check

    def _validate(self, frames_per_buffer):
        assert hasattr(self, "pa"), \
            "Call open() or use the 'with' statement before using play(), rec() or play_rec()"

        assert self._check_pow2(frames_per_buffer), \
            "Use a buffer size that is a power of 2 (1024, 2048, 4096, ...)"

        return True

    def _get_missing_frames(self, frames_per_buffer, length):
        """Calculate the number of frames missing to fill a buffer"""
        missing_frames = frames_per_buffer - (length%frames_per_buffer)

        self._logger.debug("frames per buffer : %i" %frames_per_buffer)
        self._logger.debug("missing frames    : %i" %missing_frames)

        return missing_frames


    def play(self, x, frames_per_buffer=1024):
        """Play audio. If dropouts or buffer underruns occur try different
        values for the frames_per_buffer variable."""

        _Device.play(self, x)
        self._validate(frames_per_buffer)

        missing_frames = self._get_missing_frames(frames_per_buffer, len(x))

        # generate silence to fill up missing frames
        pad = Audio(channels=x.ch, fs=x.fs, nofsamples=missing_frames, dtype=x.samples.dtype)

        # append the missing frames to a copy of the audio to be played. We now have
        # audio that can be split into complete (full) buffers
        cpy = Audio(fs=x.fs, initialdata=x.samples)
        cpy.concat(pad)

        assert len(cpy)%frames_per_buffer == 0

        stream = self.pa.open(format                = self._data_format(x),
                              channels              = x.ch,
                              rate                  = x.fs,
                              frames_per_buffer     = frames_per_buffer,
                              output_device_index   = self._index_out,
                              input                 = False,
                              output                = True,
                              )
        try:
            self._logger.info("play: start")
            counter = 0

            # split the audio into chunks the size of one buffer, so we can
            # iterate over the audio in chunksizes of the same size as one buffer
            it = iter(np.split(cpy.samples, len(cpy)/frames_per_buffer))
            try:
                while True:
                    chunk = next(it)
                    stream.write(chunk.tostring(), num_frames=frames_per_buffer)
                    counter += 1

            except StopIteration:
                pass

            finally:
                stream.stop_stream()

            self._logger.debug("chunks played  : %i"    %counter)
            self._logger.debug("samples played : %i"    %(counter*frames_per_buffer))
            self._logger.debug("duration       : %.3f"  %(counter*frames_per_buffer/x.fs))

        finally:
            self._logger.debug("play: close stream")
            stream.close()

        self._logger.info("play: done")

    def play_rec(self, x, frames_per_buffer=1024):
        """Play audio and record from input. If dropouts or buffer underruns occur
        try different values for the frames_per_buffer variable."""

        _Device.play_rec(self, x)
        self._validate(frames_per_buffer)

        missing_frames = self._get_missing_frames(frames_per_buffer, len(x))

        # generate silence to fill up missing frames
        pad = Audio(channels=x.ch, fs=x.fs, nofsamples=missing_frames, dtype=x.samples.dtype)

        # append the missing frames to a copy of the audio to be played. We now have
        # audio that can be split into complete (full) buffers
        cpy = Audio(fs=x.fs, initialdata=x.samples)
        cpy.concat(pad)

        assert len(cpy)%frames_per_buffer == 0

        rec = Audio(channels=cpy.ch, fs=cpy.fs, nofsamples=len(cpy), dtype=cpy.samples.dtype)

        stream = self.pa.open(format                = self._data_format(x),
                              channels              = x.ch,
                              rate                  = x.fs,
                              frames_per_buffer     = frames_per_buffer,
                              input_device_index    = self._index_in,
                              output_device_index   = self._index_out,
                              input                 = True,
                              output                = True,
                              )
        try:
            self._logger.info("play_rec: start")
            counter = 0

            # split the audio into chunks the size of one buffer, so we can
            # iterate over the audio in chunksizes of the same size as one buffer
            it_out = iter(np.split(cpy.samples, len(cpy)/frames_per_buffer))
            it_in  = iter(np.split(rec.samples, len(rec)/frames_per_buffer))
            try:
                while True:
                    chunk_out   = next(it_out)
                    chunk_in    = next(it_in)

                    stream.write(chunk_out.tostring(), num_frames=frames_per_buffer)

                    raw_1d      = np.fromstring(stream.read(frames_per_buffer),
                                                dtype=rec.samples.dtype)
                    # because we use an iterator chunk_in is a sliding window in the rec variable
                    chunk_in[:] = raw_1d.reshape((frames_per_buffer, rec.ch))

                    counter += 1

            except StopIteration:
                pass

            finally:
                stream.stop_stream()

            self._logger.debug("chunks played  : %i"    %counter)
            self._logger.debug("samples played : %i"    %(counter*frames_per_buffer))
            self._logger.debug("duration       : %.3f"  %(counter*frames_per_buffer/x.fs))

        finally:
            self._logger.debug("play_rec: close stream")
            stream.close()

        # remove the padding (empty frames) added to fill the last buffer. Trim
        # at the start, since we can treat that as latency.
        rec.trim(start=missing_frames, end=None)

        self._logger.debug("play_rec: trimmed %i samples from the start" %missing_frames)
        self._check_if_clipped(rec)
        self._logger.info("play_rec: done")

        return rec

    def rec(self, duration=None, channels=1, fs=96000, frames_per_buffer=1024, dtype=np.float32):
        """Record. If dropouts or buffer underruns occur try different
        values for the frames_per_buffer variable."""

        _Device.rec(self, duration=duration, channels=channels, fs=fs)
        self._validate(frames_per_buffer)

        missing_frames = self._get_missing_frames(frames_per_buffer, int(duration*fs))

        nofsamples = missing_frames+int(duration*fs)

        rec = Audio(channels=channels, fs=fs, nofsamples=nofsamples, dtype=dtype)

        assert len(rec)%frames_per_buffer == 0

        stream = self.pa.open(format                = self._data_format(rec),
                              channels              = rec.ch,
                              rate                  = rec.fs,
                              frames_per_buffer     = frames_per_buffer,
                              input_device_index    = self._index_in,
                              input                 = True,
                              output                = False,
                              )
        try:
            self._logger.info("rec: start")
            counter = 0

            # split the audio into chunks the size of one buffer, so we can
            # iterate over the audio in chunksizes of the same size as one buffer
            it_in = iter(np.split(rec.samples, len(rec)/frames_per_buffer))
            try:
                while True:
                    chunk_in    = next(it_in)
                    raw_1d      = np.fromstring(stream.read(frames_per_buffer),
                                                dtype=rec.samples.dtype)
                    # because we use an iterator chunk_in is a sliding window in the rec variable
                    chunk_in[:] = raw_1d.reshape((frames_per_buffer, rec.ch))

                    counter += 1

            except StopIteration:
                pass

            finally:
                stream.stop_stream()

            self._logger.debug("chunks recorded : %i" %counter)
            self._logger.debug("samples recorded: %i" %(counter*frames_per_buffer))
            self._logger.debug("duration        : %.3f" %(counter*frames_per_buffer/rec.fs))

        finally:
            self._logger.debug("rec: close stream")
            stream.close()

        # remove the padding (empty frames) added to fill the last buffer. Trim
        # at the start, since we can treat that as latency.
        rec.trim(start=missing_frames, end=None)

        self._logger.debug("rec: trimmed %i samples from the start" %missing_frames)
        self._check_if_clipped(rec)
        self._logger.info("rec: done")

        return rec

    def _check_if_clipped(self, rec):
        """check if the recording clipped, log the first clip for each channel"""

        clipped = False

        if np.issubdtype(rec.samples.dtype, np.floating):
            max_possible_positive_value = 1.0
        else:
            # integers used.
            # get the size of the integer type used, in bytes (2 for 16bit, 4 for 32bit)
            dt = np.dtype(rec.samples.dtype)

            # calculate the maximum possible postitive value. The maximum negative
            # value is max_possible_positive_value+1 (two's complement)
            max_possible_positive_value = 2**((8*dt.itemsize)-1) - 1

        self._logger.debug("maximum possible positive value: %i" %max_possible_positive_value)

        for i, peaks in enumerate(zip(rec.peak()[0], rec.peak()[1])):
            peak_val, peak_pos = peaks
            # abs(-32768) overflows in signed 16 bit, use long(...) in py2 to get a bigger data type
            if abs(int(peak_val)) >= max_possible_positive_value:
                clipped = True
                clip_position = peak_pos/rec.fs
                self._logger.warn("channel %02i clipped at %.3f" %(i+1, clip_position))

        return clipped

__all__ = [
           'list_devices',
           'PA',
           'Stub',
           ]

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG',
                        )
    print(list_devices())
    print('++ End of script ++')
