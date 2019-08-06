"""
This is the zignal library

@author: Ronny Andersson (ronny@andersson.tk)
@copyright: (c) 2013 Ronny Andersson
@license: MIT
"""

__version__ = "0.3.0"

from .audio import *
from . import filters
from . import measure
from . import music
from . import sndcard

__all__ = [
           'filters',
           'measure',
           'music',
           'sndcard',
           ]
__all__.extend(audio.__all__)       #@UndefinedVariable
