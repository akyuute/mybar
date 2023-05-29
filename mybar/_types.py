__all__ = (

    'ColorEscaping',
    'FieldSpec',
    'BarSpec',
    'BarTemplateSpec',

    'Separator',
    'PTY_Separator',
    'TTY_Separator',
    'Line',
    'ConsoleControlCode',
    'JSONText',

    'Contents',
    'FieldName',
    'FieldOrder',
    'FormatStr',
    'Icon',
    'Pattern',
    'PTY_Icon',
    'TTY_Icon',

    'OptName',
    'OptSpec',
    'AssignmentOption',

    'Args',
    'Kwargs',

    'Duration',
    'FormatterLiteral',
    'FormatterFname',
    'FormatterFormatSpec',
    'FormatterConversion',
    'FormatterFieldSig',
    'FmtStrStructure',

    'NmConnIDSpecifier',
    'NmConnFilterSpec',

)


from collections.abc import Callable, Sequence
from enum import Enum, IntEnum
from string import Formatter
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
from os import PathLike


Bar = TypeVar('Bar')
P = ParamSpec('P')


# Used by Bar and bar.Template:
Separator: TypeAlias = str
PTY_Separator: TypeAlias = str
TTY_Separator: TypeAlias = str
Line: TypeAlias = str
ConsoleControlCode: TypeAlias = str
JSONText: TypeAlias = str

# Used by Field and Bar:
Contents: TypeAlias = str
FieldName: TypeAlias = str
FieldOrder: TypeAlias = tuple[FieldName]
FormatStr: TypeAlias = str
Icon: TypeAlias = str
Pattern: TypeAlias = str
PTY_Icon: TypeAlias = str
TTY_Icon: TypeAlias = str

# Used by cli.OptionsAsker:
OptName: TypeAlias = str
OptSpec: TypeAlias = dict[OptName, Any]
from re import Pattern
AssignmentOption: TypeAlias = Pattern[r'(?P<key>\w+)=(?P<val>.*)']


Args: TypeAlias = list
Kwargs: TypeAlias = dict


# Used by field_funcs:
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


### Format String Wonderland ###

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


class FormatStringError(ValueError):
    '''
    Base class for exceptions involving format strings.
    '''
    pass

class BrokenFormatStringError(FormatStringError):
    '''
    Raised when a format string cannot be properly parsed or contains
    positional fields (``'{}'``).
    '''
    pass


FormatterFieldSig = TypeVar('FormatterFieldSig')
class MissingFieldnameError(FormatStringError):
    '''
    Raised when a format string field lacks a fieldname (i.e. is positional)
    when one is expected.

    '''
    @classmethod
    def with_highlighting(
        cls,
        sigs: FormatterFieldSig,
        epilogue: str
    ) -> Self:
        '''
        '''
        rebuilt = ""
        highlight = " "  # Account for the repr quotation mark.

        for sig in sigs:
            field = sig.as_string()
            rebuilt += field

            if sig.name == '':  # For each positional field...
                # Skip over the part not in braces:
                highlight += " " * len(sig.lit)

                # Only highlight the part in braces.
                bad_field_len = len(field) - len(sig.lit)
                highlight += "^" * bad_field_len

            else:
                # Skip the whole field.
                highlight += " " * len(field)

        err = '\n'.join((
            "",
            "The following fields are missing fieldnames:",
            repr(rebuilt),
            highlight,
            epilogue
        ))

        return cls(err)


FmtStrStructure: TypeAlias = tuple[FormatterFieldSig]
'''
:class:`tuple[formatting.FormatterFieldSig]`

The structure of a whole format string as broken up
by :func:`string.Formatter.parse()`
'''


NmConnIDSpecifier: TypeAlias = Literal['id', 'uuid', 'path', 'apath']
'''One of several keywords NetworkManager provides to narrow down connection results.
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
'''A dict passed to :func:`get_net_stats()` to filter multiple connections.
'''


class ColorEscaping(Enum):
    ANSI = 'ANSI'
    POLYBAR = 'POLYBAR'


class FieldSpec(TypedDict, total=False):
    '''A dict representation of :class:`mybar.Field` constructor parameters.'''
    name: Required[FieldName]
    func: Callable[P, str]
    icon: Icon
    template: FormatStr
    interval: float
    align_to_seconds: bool
    overrides_refresh: bool
    threaded: bool
    always_show_icon: bool
    run_once: bool
    constant_output: str
    bar: Bar
    args: Args
    kwargs: Kwargs
    # setup: Callable[P, Kwargs]
    setup: Callable[P, P.kwargs]
    # Set this to use different icons for different output streams:
    icons: Sequence[PTY_Icon, TTY_Icon]


class BarSpec(TypedDict, total=False):
    '''A dict representation of :class:`mybar.Bar` constructor parameters.'''
    refresh_rate: float
    run_once: bool
    align_to_seconds: bool
    join_empty_fields: bool
    override_cooldown: float
    thread_cooldown: float

    # The following field params are mutually exclusive with `template`.
    field_order: Required[list[FieldName]]
    field_definitions: dict[FieldName, FieldSpec]
    separator: Separator
    separators: Sequence[PTY_Separator, TTY_Separator]

    # The `template` param is mutually exclusive with all field params.
    template: FormatStr


class BarConfigSpec(BarSpec, total=False):
    '''A dict representation of :class:`mybar.bar.BarConfig` constructor parameters.'''
    config_file: PathLike
    debug: bool
    field_icons: dict[FieldName, Icon]

