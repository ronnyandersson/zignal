'''
Created on 13 Apr 2020

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2020 Ronny Andersson
@license: MIT
'''

# Standard library
import logging

# Internal
import zignal

if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
        level='DEBUG',
        )
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    fs = 200
    a = zignal.Audio(fs=fs)
    a1 = zignal.Sinetone(f0=1, fs=fs, duration=1, gaindb=-6)
    a2 = zignal.Sinetone(f0=2, fs=fs, duration=1, gaindb=-6)
    a.append(a1, a2)
    print(a)

    a.plot(ch="all", marker="X")
    a.decimate(10)
    print(a)
    a.plot(ch="all", marker="X")

    print('-- Done --')
