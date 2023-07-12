__all__ = (
    'CONFIG_FILE',
    'DEBUG',
    'CSI',
    'CLEAR_LINE',
    'HIDE_CURSOR',
    'UNHIDE_CURSOR',
    'FONTAWESOME_ICONS',
    'USING_FONTAWESOME'
)


from ._types import ConsoleControlCode
import os.path


CONFIG_FILE: str = os.path.abspath(os.path.expanduser(
    # '~/.config/mybar/conf.json'
    '~/.config/mybar/mybar.conf'
))
'''The default bar config file path.'''

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


FONTAWESOME_ICONS = {
    'uptime': '',
    'cpu_usage': '',
    'cpu_temp': '',
    'mem_usage': '',
    'disk_usage': '',
    'battery': '',
    'net_stats': '',
}

USING_FONTAWESOME = False

