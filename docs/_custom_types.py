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
from typing import NewType, ParamSpec, TypeAlias, TypeVar


POWERS_OF_1024 = {
    'K': 1024**1,
    'M': 1024**2,
    'G': 1024**3,
    'T': 1024**4,
    'P': 1024**5,
}


ASCII_Icon = NewType('ASCII_Icon', str)
'''
'''

ASCII_Separator = NewType('ASCII_Separator', str)
'''
'''

Args = NewType('Args', list)
'''
'''

AssignmentOption = NewType(
    'AssignmentOption',
    re_Pattern[r'(?P<key>\w+)=(?P<val>.*)']
)
'''
'''

Bar = TypeVar('Bar')
BarConfig = TypeVar('BarConfig')
ConsoleControlCode = NewType('ConsoleControlCode', str)
'''
'''

Contents = NewType('Contents', str)
'''
'''


Duration = NewType('Duration', Literal[
    'secs',
    'mins',
    'hours',
    'days',
    'weeks',
    'months',
    'years'
])
'''Possible names for units of elapsed time.'''

Field = TypeVar('Field')
P = ParamSpec('P')
FieldFuncSetup = NewType('FieldFuncSetup', Callable[P, P.kwargs])
'''
'''

FieldFunc = NewType('FieldFunc', Callable[P, str])
'''
'''

FieldName = NewType('FieldName', str)
'''
'''

FieldName = NewType('FieldName', str)
'''
'''

FieldOrder = NewType('FieldOrder', tuple[FieldName])
'''
'''

FormatterFieldSig = TypeVar('FormatterFieldSig')
FieldPrecursor = FieldName | Field | FormatterFieldSig
'''
'''

FileContents = NewType('FileContents', str)
'''
'''

FormatStr = NewType('FormatStr', str)
'''
'''


FormatterConversion = NewType('FormatterConversion', str | None)
'''
The `conversion` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

FormatterFname = NewType('FormatterFname', str | None)
'''
The `field_name` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

FormatterFormatSpec = NewType('FormatterFormatSpec', str | None)
'''
The `format_spec` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

FormatterLiteral = NewType('FormatterLiteral', str | None)
'''
The `literal_text` part of one tuple in the iterable returned
by :func:`string.Formatter.parse()`.
'''

HostOption = NewType('HostOption', Literal[
    'nodename',
    'sysname',
    'release',
    'version',
    'machine',
])

Icon = NewType('Icon', str)
'''
'''

JSONText = NewType('JSONText', str)
'''
'''

Kwargs = NewType('Kwargs', dict)
'''
'''

Line = NewType('Line', str)
MetricSymbol = NewType('MetricSymbol', Literal[*POWERS_OF_1024.keys()])
'''
'''


NmConnIDSpecifier = NewType(
    'NmConnIDSpecifier', 
    Literal['id', 'uuid', 'path', 'apath']
)
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

NmConnFilterSpec = NewType('NmConnFilterSpec', Mapping[NmConnIDSpecifier, str])
'''
A mapping passed to :func:`mybar.field_funcs.get_net_stats` to filter
multiple connections.
'''

OptName = NewType('OptName', str)
'''
'''

OptSpec = NewType('OptSpec', dict[OptName, Any])
'''
'''

Pattern = NewType('Pattern', str)
'''
'''

PythonData = NewType('PythonData', object)
'''
'''

Separator = NewType('Separator', str)
'''
'''

StorageMeasure = NewType(
    'StorageMeasure',
    Literal['total', 'used', 'free', 'percent']
)
'''
'''

Unicode_Icon = NewType('Unicode_Icon', str)
'''
'''

Unicode_Separator = NewType('Unicode_Separator', str)
'''
'''


### Used in the future by field_funcs:
##class Context(NamedTuple):
##    '''
##    For future use.
##    '''
##    contents: str = None
##    state: Any = None
##
##
##class BatteryStates(Enum):
##    '''
##    For future use.
##    '''
##    CHARGING = 'charging'
##    DISCHARGING = 'discharging'
##    # Progressive/dynamic battery icons!
##        # 
##        # 
##        # 
##        # 
##        # 
##
##
##class ColorEscaping(Enum):
##    '''
##    For future use.
##    '''
##    ANSI = 'ANSI'
##    POLYBAR = 'POLYBAR'


