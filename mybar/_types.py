__all__ = (

    'ASCII_Icon',
    'ASCII_Separator',
    'Args',
    'AssignmentOption',
    'Bar',
    'BarConfig',
    # 'BatteryStates',
    # 'ColorEscaping',
    'ConsoleControlCode',
    'Contents',
    # 'Context',
    'Duration',
    'Field',
    'FieldFuncSetup',
    'FieldFunc',
    'FieldName',
    'FieldName',
    'FieldOrder',
    'FieldPrecursor',
    'FileContents',
    'FormatStr',
    'FormatterFieldSig',
    'FormatterConversion',
    'FormatterFname',
    'FormatterFormatSpec',
    'FormatterLiteral',
    'HostOption',
    'Icon',
    'JSONText',
    'Kwargs',
    'Line',
    'MetricSymbol',
    'NmConnIDSpecifier',
    'NmConnFilterSpec',
    'OptName',
    'OptSpec',
    'Pattern',
    'PythonData',
    'Separator',
    'StorageMeasure',
    'Unicode_Icon',
    'Unicode_Separator',

)


from collections.abc import Callable, Mapping
from enum import Enum
from os import PathLike
from re import Pattern as re_Pattern
from typing import (
    Any,
    Literal,
    NamedTuple,
    Required,
)


POWERS_OF_1024 = {
    'K': 1024**1,
    'M': 1024**2,
    'G': 1024**3,
    'T': 1024**4,
    'P': 1024**5,
}


type ASCII_Icon = str
type ASCII_Separator = str
type Args = list
type AssignmentOption = re_Pattern[r'(?P<key>\w+)=(?P<val>.*)']
type Bar = 'Bar'
type BarConfig = 'BarConfig'
type ConsoleControlCode = str
type Contents = str

type Duration = Literal[
    'secs',
    'mins',
    'hours',
    'days',
    'weeks',
    'months',
    'years'
]
'''Possible names for units of elapsed time.'''

type Field = 'Field'
type FieldFuncSetup[**P] = Callable[P, P.kwargs]
type FieldFunc[**P] = Callable[P, str]
type FieldName = str
type FieldName = str
type FieldOrder = tuple[FieldName]
type FieldPrecursor = FieldName | Field | FormatterFieldSig
type FileContents = str
type FormatStr = str
type FormatterFieldSig = 'FormatterFieldSig'

type FormatterConversion = str | None
'''
The `conversion` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

type FormatterFname = str | None
'''
The `field_name` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

type FormatterFormatSpec = str | None
'''
The `format_spec` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

type FormatterLiteral = str | None
'''
The `literal_text` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

type HostOption = Literal[
    'nodename',
    'sysname',
    'release',
    'version',
    'machine',
]

type Icon = str
type JSONText = str
type Kwargs = dict[str]
type Line = str
type MetricSymbol = Literal[*POWERS_OF_1024.keys()]

type NmConnIDSpecifier = Literal['id', 'uuid', 'path', 'apath']
'''
One of several keywords NetworkManager provides to narrow down
connection results. From the ``nmcli`` man page:

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

type NmConnFilterSpec = Mapping[NmConnIDSpecifier, str]
'''
A mapping passed to :func:`mybar.field_funcs.get_net_stats` to filter
multiple connections.
'''

type OptName = str
type OptSpec = dict[OptName, Any]
type Pattern = str
type PythonData = object
type Separator = str
type StorageMeasure = Literal['total', 'used', 'free', 'percent']
type Unicode_Icon = str
type Unicode_Separator = str


# Used in the future by field_funcs:
class Context(NamedTuple):
    '''
    For future use.
    '''
    contents: str = None
    state: Any = None


class BatteryStates(Enum):
    '''
    For future use.
    '''
    CHARGING = 'charging'
    DISCHARGING = 'discharging'
    # Progressive/dynamic battery icons!
        # 
        # 
        # 
        # 
        # 


class ColorEscaping(Enum):
    '''
    For future use.
    '''
    ANSI = 'ANSI'
    POLYBAR = 'POLYBAR'


