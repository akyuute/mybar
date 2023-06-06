"""
######
mybar
######

*Craft highly customizable status bars with ease.*

**mybar** is a versatile status bar library and tool written in Python.


:copyright: (c) 2021-present by LonelyAbsol.
:license: MIT, see LICENSE for more details.
"""

__title__ = 'mybar'
__description__ = "An async status bar with a highly customizable API."
__url__ = "https://github.com/lonelyabsol/mybar"
__version__ = '0.7'
__author__ = "LonelyAbsol"
__license__ = 'MIT'
__copyright__ = "Copyright (c) 2021-present LonelyAbsol"


from .constants import *
from . import utils
from .bar import *
from .field import *

