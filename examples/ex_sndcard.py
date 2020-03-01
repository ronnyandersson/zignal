'''
Created on 16 Feb 2015

This example will play some audio on the system standard sound card.

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Internal
import zignal.sndcard


def ex_1_play():
    # The recommended way of creating and using a sndcard instance is by using the
    # "with" statement. This will make sure that the instance is closed correctly
    # after usage. See http://effbot.org/zone/python-with-statement.htm
    #
    # This example plays the audio on the default device

    fs = 44100
    x  = zignal.Sinetone(f0=400, fs=fs, duration=1.5, gaindb=-12)
    x2 = zignal.Sinetone(f0=900, fs=fs, duration=1.5, gaindb=-18)

    x.append(x2)
    x.convert_to_float(targetbits=32)

    with zignal.sndcard.PA(device_in='default', device_out='default') as snd:
        # using an assert here helps PyDev in eclipse when pressing ctrl+space for autocomplete
        assert isinstance(snd, zignal.sndcard._Device)
        snd.play(x)


def ex_2_play():
    # Another way of using a sndcard is by first creating an instance and
    # manually calling the open() function. The close() function *must* be
    # called in a controlled fashion. This usually means that the usage is
    # wrapped in a try-except-finally clause.

    fs = 44100
    x  = zignal.Sinetone(f0=700, fs=fs, duration=1.0, gaindb=-24)
    xn = zignal.Noise(channels=1, fs=fs, duration=1.0, gaindb=-12, colour='pink')
    x.append(xn)
    x.convert_to_integer(targetbits=16)

    snd = zignal.sndcard.PA()
    print(snd)

    snd.open()
    try:
        snd.play(x)
    finally:
        snd.close()


def ex_3_play_rec():
    # Play and record at the same time

    fs = 44100
    x  = zignal.Sinetone(f0=500, fs=fs, duration=1.5, gaindb=-12)
    x.convert_to_float(targetbits=32)

    with zignal.sndcard.PA(device_in='default', device_out='default') as snd:
        # using an assert here helps PyDev in eclipse when pressing ctrl+space for autocomplete
        assert isinstance(snd, (zignal.sndcard.PA))
        y = snd.play_rec(x, frames_per_buffer=32)

    print(y)
    y.plot()


def ex_4_rec():
    # Record

    fs = 44100
    with zignal.sndcard.PA(device_in='default') as snd:
        # using an assert here helps PyDev in eclipse when pressing ctrl+space for autocomplete
        assert isinstance(snd, (zignal.sndcard.PA))
        print("recording...")
        y = snd.rec(duration=3.5, channels=1, fs=fs)

    print(y)
    y.plot()


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    ex_1_play()
    ex_2_play()
    ex_3_play_rec()
    ex_4_rec()

    print('++ End of script ++')
