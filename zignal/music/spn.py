'''
Created on 21 Feb 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT

Scientific pitch notation (also known as American Standard Pitch Notation
(ASPN) and International Pitch Notation (IPN))
'''

# standard library
import logging
import re

def key2index(key='A4'):
    """ Scientific pitch notation key mapped to an index

    Example:

       >>> key2index('A4')
       49

    A0 is 1, C1 is 4, A4 is 49

    Double sharps are notated with 'x'

    https://en.wikipedia.org/wiki/Scientific_pitch_notation
    https://en.wikipedia.org/wiki/Sharp_(music)
    https://en.wikipedia.org/wiki/Flat_(music)
    """

    logger = logging.getLogger(__name__)

    valid_chars = "ABCDEFGb#x-0123456789"

    for c in key:
        assert c in valid_chars, "Valid characters are: '%s'" %valid_chars

    m = re.match(r"(?P<note>^[A-G])(?P<half>b{0,3}|#{0,1}x{0,1})(?P<octave>[\-]?[0-9]+)", key)
    if m is None:
        raise ValueError("Failed to match key '%s'" %key)

    matched = m.groupdict()

    logger.debug("key: %s re: %s" %(key.ljust(4), matched))

    octave  = int(matched.get('octave'))
    note    = matched.get('note')
    half    = matched.get('half')

    aug_or_dim = {
                  'bbb' : -3,
                  'bb'  : -2,
                  'b'   : -1,
                  '#'   :  1,
                  'x'   :  2,
                  '#x'  :  3,
                  }

    octave  = (octave-1)*12
    note    = " C D EF G A B".index(note)
    #          ^ ^ ^  ^ ^ ^ the spaces are important, we can 'index-search' the array
    half    = aug_or_dim.get(half, 0)

    # A0      = 1
    # A#0/Bb0 = 2
    # B0      = 3
    # C1      = 4
    pitch = 3 + octave + note + half

    logger.debug("key: %s spn: %i" %(key.ljust(4), pitch))

    return pitch

def index2key(index=49):
    """Given the SPN index, return the corresponding SPN key

    Example:

        >>> index2key(1)
        'A0'
        >>> index2key(4)
        'C1'
        >>> index2key(49)
        'A4'
        >>>

    """

    logger = logging.getLogger(__name__)

    assert isinstance(index, int)

    notes   = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    # shift the index around so that
    #    C0 --> -8
    #    A0 -->  1
    #    C1 -->  4
    notes_idx   = (index-3)%12 -1
    octave      = (index+8)//12

    # use the lookup table 'notes' to get the note name
    note        = notes[notes_idx]

    logger.debug("index %3i notes_idx %3i: %s%i" %(index, notes_idx, note, octave))

    spn_key = "%s%i" %(note, octave)

    return spn_key

__all__ = [
           'key2index',
           'index2key',
           ]

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    print('-- Done --')
