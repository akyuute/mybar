"""
######
mybar
######

*Craft highly customizable status bars with ease.*

**mybar** is a versatile status bar library and command line tool
written in Python.


:copyright: (c) 2021-present akyuute.
:license: MIT, see LICENSE for more details.
"""

__title__ = 'mybar'
__description__ = "Craft highly customizable status bars with ease."
__url__ = "https://github.com/akyuute/mybar"
__version__ = '0.12'
__author__ = "akyuute"
__license__ = 'MIT'
__copyright__ = "Copyright (c) 2021-present akyuute"


__all__ = (
    'Bar',
    'BarConfig',
    'CONFIG_FILE',
    'DEBUG',
    'Field',
    'run',
    'write_initial_config'
)


from os import PathLike

from .constants import CONFIG_FILE, DEBUG
from .bar import Bar, BarConfig
from .field import Field


def run(*, once: bool = False, file: PathLike = None) -> None:
    '''
    Generate a new :class:`Bar` from a config file and run it in STDOUT.
    Ask to write the file if it doesn't exist.

    :param once: Print the bar only once, defaults to ``False``
    :type once: :class:`bool`

    :param file: The config file to source,
        defaults to :obj:`CONFIG_FILE`
    :type file: :class:`PathLike`
    '''
##    import __main__ as possibly_repl
##    if not hasattr(possibly_repl, '__file__'):
##        # User is in a REPL
    try:

        try:
            bar = Bar.from_file(file)
        except FileNotFoundError as e:
            _p_ = __package__
            msg = (
                f"Try running `{_p_}.write_initial_config()`"
                f" if this is your first time using {_p_}."
            )
            e.add_note(msg)
            raise e from None

        bar.run(once=once)

    except KeyboardInterrupt:
        print()
        return


def write_initial_config() -> None:
    '''
    Write a new default config file after getting user approval.
    '''
    BarConfig.write_with_approval(CONFIG_FILE)


del PathLike

