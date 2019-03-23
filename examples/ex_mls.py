'''
Created on 7 Dec 2015

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2015 Ronny Andersson
@license: MIT
'''

# standard library
from __future__ import print_function
import logging

# custom libraries
import zignal

def fake_system(x, fs=None):
    f1 = zignal.filters.biquads.RBJ(filtertype="peak", gaindb=-30, f0=10,   Q=0.707*10, fs=fs)
    f2 = zignal.filters.biquads.RBJ(filtertype="peak", gaindb=50,  f0=100,  Q=0.707*10, fs=fs)
    f3 = zignal.filters.biquads.RBJ(filtertype="peak", gaindb=-60, f0=1000, Q=0.707*10, fs=fs)

    y = zignal.Audio(fs=fs, initialdata=x)
    y.samples = f1.filter_samples(y.samples)
    y.samples = f2.filter_samples(y.samples)
    y.samples = f3.filter_samples(y.samples)

    y.delay(24000)

    return y

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)-7s: %(module)s.%(funcName)-15s %(message)s',
                        level='DEBUG')

    # sample rate
    fs = 48000

    # mls order. Relates to the sample rate and the length of the impulse we want to capture
    N = 16

    # The number of repeated mls sequences. More repetitions gives a better S/N ratio
    rep = 5

    # An emphasis filter. Here we put more energy into the low frequencies to stimulate
    # the systems response in the lower frequency area, giving a better frequency response
    f = zignal.filters.biquads.RBJ(filtertype="highshelf", gaindb=-10, f0=100, Q=0.707, fs=fs)
    B, A = f.get_coefficients()

    # Get a set of taps related to the mls order N. Either use the same all the time
    # or select a random set at each run.
    taps = zignal.measure.mls.TAPS[N][0]
    #taps = zignal.measure.mls.get_random_taps(N)

    # Create the mls instance and its full sequence.
    mls = zignal.measure.mls.MLS(N=N, taps=taps, fs=fs, repeats=rep, B=B, A=A)

    print (repr(mls))
    print(mls)

    mls.plot(label="mls signal unfiltered")

    # apply the emphasis
    mls.apply_emphasis()

    mls.plot(label="mls signal post emphasis")

    # send the signal through a system
    y = fake_system(mls.samples, fs=fs)
    y.plot(label="signal post system")

    # apply deemphasis before we extract the impulse response. Flip comments
    # to "forget" to apply the deemphasis
    z = mls.apply_deemphasis(y.samples)
    #z = y.samples

    # extract the impulse response. Will throw away the first sequence and average the others
    k = mls.get_impulse(z)
    k.plot(label="impulse in time domain")
    k.plot_fft(window='rectangular', normalise=False)

    # Note how the lowest filter in the system (the one at 10 Hz) is barely detected. We
    # can also see that the impulse in the time domain hasn't finished ringing during the
    # time. This is all related. Increasing the MLS order N will capture a longer impulse
    # and give better performace at lower frequencies. This is always a tradeoff between
    # performance and resolution.
    #
    # The impulse also wraps around when plotted in the time domain. This is not a
    # problem when we are interested in the magnitude response. If we are interested in
    # the phase response the delay between time=0 and the "start" of the impulse will
    # determine the amount of phase unwrapping we will have to apply.
