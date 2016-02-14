# zignal

This is a python audio signal processing library.


## Basic example

    import zignal
    
    x = zignal.Sinetone(fs=44100, f0=997, duration=0.01, gaindb=-20)
    print(x)
    x.plot()

See the examples folder for more examples.

## Requirements

This library relies on numpy, scipy and matplotlib. At the moment it is
recommended to install these using the systems default package manager.
On debian/ubuntu, do a 

    sudo apt-get install python-numpy python-scipy python-matplotlib python-pyaudio

to install the requirements.

## Design goals

1.  Readability over efficiency. This is a python library for development and
    understanding of audio signal processing.
2.  The initial goal is to write the functionality in pure python, with the
    use of numpy, scipy and matplotlib. See rule 1. If efficiency becomes an
    issue a c/c++ library might be implemented but the pure python code must
    remain the default choice.
3.  Design for non real-time processing. Functionality to do real-time
    processing can be added if it does not break rule 1.
4.  Self documentation. The code should aim to be well documented, in the
    source code itself.
