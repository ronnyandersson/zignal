"""
This is the zignal library

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2013 Ronny Andersson
@license: MIT
"""

__version__ = "0.6.0"

# must be imported first
from .audio import *    # isort:skip

# Local folders
from . import filters, measure, music, sndcard

__all__ = [
    'filters',
    'measure',
    'music',
    'sndcard',
    ]
__all__.extend(audio.__all__)       #@UndefinedVariable
