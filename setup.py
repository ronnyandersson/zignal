from setuptools import setup, find_packages
import os
import ast

# Dynamically calculate the version
version = None
with open(os.path.join('zignal', '__init__.py'), 'r') as fp:
    for line in fp:
        if line.startswith('__version__'):
            version = ast.parse(line).body[0].value.s
            break

assert version is not None, "Problem reading the version from __init__.py"

setup(
    name                = 'zignal',
    author              = 'Ronny Andersson',
    author_email        = 'ronny@andersson.tk',
    packages            = find_packages(),
    install_requires    = [
                           'numpy',
                           'scipy',
                           'matplotlib',
                           ],
    extras_require      = {
                           'sndcard': ['pyaudio'],
                           'testing': ['nose'],
                          },
    url                 = 'https://github.com/ronnyandersson/zignal',
    download_url        = 'https://pypi.python.org/pypi/zignal',
    license             = open('LICENSE.txt').read(),
    version             = version,
    description         = 'Audio signal processing library',
    long_description    = open('README.md').read(),
    platforms           = ['any'],
    entry_points        = {
                           'console_scripts': \
                           [
                            'zignal-listsndcards=zignal.sndcard:list_devices',
                            ],
                           },
    keywords            = [
                           'audio',
                           'sound',
                           'card',
                           'soundcard',
                           'pyaudio',
                           'portaudio',
                           'playback',
                           'recording',
                           'digital',
                           'signal',
                           'processing',
                           'DSP',
                           'signalprocessing',
                           'fourier',
                           'FFT',
                           'filter',
                           'filtering',
                           'parametric',
                           'eq',
                           'equaliser',
                           'equalizer',
                           'biquad',
                           'cookbook',
                           'sine',
                           'generator',
                           'mls',
                           'mlssa',
                           'maximum',
                           'length',
                           'sequence',
                           'maximumlengthsequence',
                           'pseudo',
                           'random',
                           'pseudorandom',
                           'measure',
                           'measurement',
                           'impulse',
                           'response',
                           'impulseresponse',
                           'frequency',
                           'frequencyresponse',
                           'magnitude',
                           'magnituderesponse',
                           'piano',
                           'midi',
                           'tuning',
                           'scale',
                           'pitch',
                           'notation',
                           'equal',
                           'temperament',
                           '12TET',
                           'spn',
                           ],
    classifiers         = [
                           'Development Status :: 3 - Alpha',
                           'Environment :: Console',
                           'Intended Audience :: Developers',
                           'Intended Audience :: Education',
                           'Intended Audience :: Science/Research',
                           'License :: OSI Approved :: MIT License',
                           'Operating System :: OS Independent',
                           'Programming Language :: Python',
                           'Programming Language :: Python :: 3.6',
                           'Topic :: Education',
                           'Topic :: Multimedia :: Sound/Audio :: Analysis',
                           'Topic :: Multimedia :: Sound/Audio :: Capture/Recording',
                           'Topic :: Multimedia :: Sound/Audio :: Editors',
                           'Topic :: Multimedia :: Sound/Audio :: MIDI',
                           'Topic :: Multimedia :: Sound/Audio :: Mixers',
                           'Topic :: Multimedia :: Sound/Audio :: Players',
                           'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
                           'Topic :: Scientific/Engineering :: Mathematics',
                           'Topic :: Software Development :: Quality Assurance',
                           'Topic :: Software Development :: Testing',
                           ],
)
