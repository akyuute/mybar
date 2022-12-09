"""
mybar
~~~~~

An asynchronous status bar with an intuitive, highly customizable API.

:copyright: (c) 2022 by SleepyAbsol.
:license: MIT, see LICENSE for more details.
"""

__title__ = 'mybar'
__description__ = "An async status bar with a highly customizable API."
__url__ = "https://github.com/sleepyabsol/mybar"
__version__ = '0.3'
__author__ = "SleepyAbsol"
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2021-2022 SleepyAbsol'


from . import cli
from . import field_funcs
from . import setups
from . import utils
from .base import *
from .errors import *

