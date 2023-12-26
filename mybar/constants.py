__all__ = (
    'CONFIG_FILE',
    'DEBUG',
    'CSI',
    'CLEAR_LINE',
    'HIDE_CURSOR',
    'UNHIDE_CURSOR',
)


import sys
import os.path

from ._types import ConsoleControlCode


CONFIG_FILE: str = (
    '~/.config/mybar/mybar.conf'
    # '~/.config/mybar/conf.json'
)
'''The default mybar config file path.'''

DEBUG: bool = False
'''The default debug state.'''


# Used by Bar:
CSI: ConsoleControlCode = '\033['
'''Unix terminal escape code (control sequence introducer).'''

CLEAR_LINE: ConsoleControlCode = '\x1b[2K'
'''VT100 escape code to clear the line.'''

HIDE_CURSOR: ConsoleControlCode = '?25l'
'''VT100 escape code to hide the cursor.'''

UNHIDE_CURSOR: ConsoleControlCode = '?25h'
'''VT100 escape code to unhide the cursor.'''

