"""
mybar
~~~~~

An asynchronous status bar with an intuitive, highly customizable API.

:copyright: (c) 2021-present by LonelyAbsol.
:license: MIT, see LICENSE for more details.
"""

__title__ = 'mybar'
__description__ = "An async status bar with a highly customizable API."
__url__ = "https://github.com/lonelyabsol/mybar"
__version__ = '0.6'
__author__ = "LonelyAbsol"
__license__ = 'MIT'
__copyright__ = "Copyright (c) 2021-present LonelyAbsol"


# CONFIG_FILE: str = '~/.mybar.json'  # The default bar config file path
# DEBUG: bool = False


from .constants import *
from . import utils
from .bar import *
from .field import *

