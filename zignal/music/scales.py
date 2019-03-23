'''
Created on 21 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# standard library
import logging

# external libraries
import numpy as np

# local libraries
from . import spn

def equal_temperament(n):
    """Twelve-tone equal temperament (12TET) divides the octave into 12
    equal parts, making the interval between two ajacent notes the twelfth
    root of two.

    The argument n can be a number or a list/tuple/iterable.

        2^(1/12):1 --> 2^(n/12)

    https://en.wikipedia.org/wiki/Equal_temperament
    """

    ratio = np.power(2, (np.array(n)/12))

    return ratio

def piano_key2freq(n, a=49, tuning=440):
    """Twelve-tone equal temperament tuning for a theoretically ideal piano.

    The argument n can be a number or a list/tuple/iterable.

    The 49th key, called A4 is tuned to the reference (tuning) frequency, normally
    440Hz. The frequency is then given by

        f(n) = 440*2^((n-49)/12)

    https://en.wikipedia.org/wiki/Piano_key_frequencies
    """

    frequency = tuning*equal_temperament(np.array(n)-a)

    return frequency

def piano_freq2key(f, a=49, tuning=440, quantise=False):
    """Frequency [f] to twelve-tone equal temperament tuning for a theoretically
    ideal piano, where 440Hz-->49
    """

    key = 12*np.log2(f/tuning) + a

    if quantise:
        key = np.int(np.round(key))

    return key

def piano_note2freq(note, tuning=440):
    """Convert a piano note like 'C4' to 12TET frequency 261.6 Hz"""
    idx  = spn.key2index(note)
    freq = piano_key2freq(idx, tuning=tuning)

    return freq

def piano_freq2note(f, tuning=440):
    """Given frequency f, calculate the nearest note in 12TET SPN notation"""
    idx = piano_freq2key(f, tuning=tuning, quantise=True)
    key = spn.index2key(idx)

    return key

def midi_key2freq(n, tuning=440):
    """MIDI Tuning Standard. Convert midi note n to frequency f [Hz]. MIDI note 69
    equals 440 Hz, equal temperament.

    http://en.wikipedia.org/wiki/MIDI_Tuning_Standard
    """

    frequency = piano_key2freq(n, a=69, tuning=tuning)

    return frequency

def midi_freq2key(f, tuning=440, quantise=False):
    """ MIDI Tuning Standard. Convert frequency f [Hz] to a midi note. MIDI note 69
    equals 440 Hz, equal temperament

    69+12*log2(f/440)

    http://en.wikipedia.org/wiki/MIDI_Tuning_Standard
    """

    midinote = piano_freq2key(f, a=69, tuning=tuning, quantise=quantise)

    return midinote

__all__ = [
           'equal_temperament',
           'piano_key2freq',
           'piano_freq2key',
           'piano_note2freq',
           'piano_freq2note',
           'midi_key2freq',
           'midi_freq2key',
           ]

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    print('-- Done --')
