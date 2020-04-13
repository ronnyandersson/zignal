'''
Created on 12 Apr 2020

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2020 Ronny Andersson
@license: MIT

Demo of how to iterate over an instance of the Audio class, for chunk-based
processing. Typically the chunks have a size that is a power of two, for
example 256, 1024 or 4096. In this example the chunk size is set to 1000
for simplicity in the plots. The sample rate in this example is also set to
a value that enhances the effect of the example, since hera a chunk equals
to one second of data.
'''

# Standard library
import logging

# Third party
import matplotlib.pyplot as plt
import numpy as np

# Internal
import zignal

if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
        level='DEBUG',
        )
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("zignal").setLevel(logging.DEBUG)

    fs = 1000

    # Create various ramp signals, to visualise the chunks better. Not real
    # audio, but shows in a plot what the chunks look like
    a1 = zignal.Audio(fs=fs, initialdata=np.linspace(0, 1,  num=(1000/2)))
    a2 = zignal.Audio(fs=fs, initialdata=np.linspace(0, -1, num=(1000*1)+500))
    a3 = zignal.Audio(fs=fs, initialdata=np.linspace(0, 1,  num=(1000*2)+200))

    a = zignal.Audio(fs=fs)
    a.append(a1, a2, a3)
    print(a)

    # We now have 2.2 seconds of audio in three channels. This does not add up
    # to even chunk sizes, so padding will have to be done in order to iterate.
    #
    # Three (3) chunks are expected.
    for val in a.iter_chunks(chunksize=1000):
        print("------------------------------------------------")
        print("shape of data in chunk: %s" % str(val.shape))
        print(val)

        plt.figure(1)
        plt.plot(val[:, 0], ls="-",  label="a1")
        plt.plot(val[:, 1], ls="--", label="a2")
        plt.plot(val[:, 2], ls="-.", label="a3")
        plt.grid()
        plt.ylim(-1.1, 1.1)
        plt.xlabel("samples in chunk")
        plt.ylabel("magnitude [lin]")
        plt.legend(loc="upper right")
        plt.show()

    # We can pad beforehand if we know how many samples are missing, then no
    # padding will occur inside the iterator
    b = a.copy()

    b.gain(-20)   # just to get a debug logging entry
    b.pad(nofsamples=800)
    print(b)
    for val in b.iter_chunks(chunksize=1000):
        print("------------------------------------------------")
        print("shape of data in chunk: %s" % str(val.shape))
        print(val)

    print('-- Done --')
