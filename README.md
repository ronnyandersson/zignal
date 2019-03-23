# zignal

This is a python audio signal processing library.


## Example usage

    >>> import zignal
    >>>
    >>> x = zignal.Sinetone(fs=44100, f0=997, duration=0.1, gaindb=-20)
    >>> print(x)
    =======================================
    classname        : Sinetone
    sample rate      : 44100.0 [Hz]
    channels         : 1
    duration         : 0.100 [s]
    datatype         : float64
    samples per ch   : 4410
    data size        : 0.034 [Mb]
    has comment      : no
    peak             : [ 0.1]
    RMS              : [ 0.0707]
    crestfactor      : [ 1.4147]
    -----------------:---------------------
    frequency        : 997.0 [Hz]
    phase            : 0.0 [deg]
    -----------------:---------------------

    >>> x.fade_out(millisec=10)
    >>> x.convert_to_float(targetbits=32)
    >>> x.write_wav_file("sinetone.wav")
    >>> x.plot()
    >>> x.plot_fft()
    >>>
    >>> f = zignal.filters.biquads.RBJ(filtertype="peak", gaindb=-6, f0=997, Q=0.707, fs=96000)
    >>> print(f)
    =======================================
    classname        : RBJ
    sample rate      : 96000.0 [Hz]
    feedforward  (B) : [ 0.96949457 -1.87369167  0.90819329]
    feedback     (A) : [ 1.         -1.87369167  0.87768787]
    number of zeros  : 2
    number of poles  : 2
    minimum phase?   : Yes
    -----------------:---------------------
    stable?          : Yes
    type             : peak
    gain             : -6.00 [dB]
    f0               : 997.0 [Hz]
    Q                : 0.7070

    >>> f.plot_mag_phase()
    >>> f.plot_pole_zero()
    >>>

See the examples folder for more examples.

## Requirements

This library relies on numpy, scipy, matplotlib and pyaudio. At the moment it is recommended to install these using the systems default package manager. On debian/ubuntu, do a

    sudo apt-get install python-numpy python-scipy python-matplotlib python-pyaudio

to install the requirements. Once the basic requirements are installed, use pip to install zignal in for example a virtualenv. Create a virtualenv that can access the system site packages,

    virtualenv --system-site-packages <name-of-virtualenv>
    . <name-of-virtualenv>/bin/activate
    pip install zignal

## Design goals

1.  Readability over efficiency. This is a python library for development and understanding of audio signal processing.
2.  The initial goal is to write the functionality in pure python, with the use of numpy, scipy and matplotlib. See rule 1. If efficiency becomes an issue a c/c++ library might be implemented but the pure python code must remain the default choice.
3.  Design for non real-time processing. Functionality to do real-time processing can be added if it does not break rule 1.
4.  Self documentation. The code should aim to be well documented, in the source code itself.
