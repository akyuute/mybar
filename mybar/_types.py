__all__ = (

    'ColorEscaping',

    'Separator',
    'PTY_Separator',
    'TTY_Separator',
    'Line',
    'ConsoleControlCode',
    'JSONText',

    'Contents',
    'Field',
    'FieldName',
    'FieldOrder',
    'FieldPrecursor',
    'Icon',
    'Pattern',
    'PTY_Icon',
    'TTY_Icon',

    'FormatStr',
    'FormatterLiteral: TypeAlias = str | None',
    'FormatterFname: TypeAlias = str | None',
    'FormatterFormatSpec: TypeAlias = str | None',
    'FormatterConversion: TypeAlias = str | None',

    'OptName',
    'OptSpec',
    'AssignmentOption',

    'Args',
    'Kwargs',

    'Duration',

    'NmConnIDSpecifier',
    'NmConnFilterSpec',

)


from collections.abc import Callable, Sequence
from enum import Enum, IntEnum
from os import PathLike
from re import Pattern as re_Pattern
from typing import (
    Any,
    Literal,
    NamedTuple,
    ParamSpec,
    Required,
    Self,
    TypeAlias,
    TypedDict,
    TypeVar
)


Field = TypeVar('Field')
FieldPrecursor = TypeVar('FieldPrecursor')
Bar = TypeVar('Bar')
P = ParamSpec('P')


Args: TypeAlias = list
Kwargs: TypeAlias = dict

# Used by Bar and BarConfigSpec:
Separator: TypeAlias = str
PTY_Separator: TypeAlias = str
TTY_Separator: TypeAlias = str
Line: TypeAlias = str
ConsoleControlCode: TypeAlias = str
JSONText: TypeAlias = str
FileContents: TypeAlias = str

# Used by Field and Bar:
Contents: TypeAlias = str
FieldName: TypeAlias = str
FieldOrder: TypeAlias = tuple[FieldName]
FormatStr: TypeAlias = str
Icon: TypeAlias = str
Pattern: TypeAlias = str
PTY_Icon: TypeAlias = str
TTY_Icon: TypeAlias = str

FormatterLiteral: TypeAlias = str | None
'''The `literal_text` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

FormatterFname: TypeAlias = str | None
'''The `field_name` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

FormatterFormatSpec: TypeAlias = str | None
'''The `format_spec` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

FormatterConversion: TypeAlias = str | None
'''The `conversion` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''


# Used by cli.OptionsAsker:
OptName: TypeAlias = str
OptSpec: TypeAlias = dict[OptName, Any]
AssignmentOption: TypeAlias = re_Pattern[r'(?P<key>\w+)=(?P<val>.*)']


# Used by field_funcs:
class Context(NamedTuple):
    contents: str = None
    state: Any = None

class BatteryStates(Enum):
    CHARGING = 'charging'
    DISCHARGING = 'discharging'
    # Progressive/dynamic battery icons!
        # 
        # 
        # 
        # 
        # 

POWERS_OF_1024 = {
    'K': 1024**1,
    'M': 1024**2,
    'G': 1024**3,
    'T': 1024**4,
    'P': 1024**5,
}

MetricSymbol = Literal[*POWERS_OF_1024.keys()]
DiskMeasure = Literal['total', 'used', 'free', 'percent']
Duration: TypeAlias = Literal[
    'secs',
    'mins',
    'hours',
    'days',
    'weeks',
    'months',
    'years'
]
'''Possible names for units of elapsed time.'''

NmConnIDSpecifier: TypeAlias = Literal['id', 'uuid', 'path', 'apath']
'''
One of several keywords NetworkManager provides to narrow down connection results.
From the ``nmcli`` man page:

.. code-block:: none

   id, uuid, path and apath keywords can be used if ID is
   ambiguous. Optional ID-specifying keywords are:

   id
       the ID denotes a connection name.

   uuid
       the ID denotes a connection UUID.

   path
       the ID denotes a D-Bus static connection path in the format of
       /org/freedesktop/NetworkManager/Settings/num or just num.

   apath
       the ID denotes a D-Bus active connection path in the format of
       /org/freedesktop/NetworkManager/ActiveConnection/num or just num.
'''

NmConnFilterSpec: TypeAlias = dict[NmConnIDSpecifier, str]
'''
A dict passed to :func:`get_net_stats()` to filter multiple connections.
'''


class ColorEscaping(Enum):
    ANSI = 'ANSI'
    POLYBAR = 'POLYBAR'


