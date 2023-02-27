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
    'FormatStr',
    'Icon',
    'Pattern',
    'PTY_Icon',
    'TTY_Icon',

    'OptName',
    'OptSpec',

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
FormatStr: TypeAlias = str
Icon: TypeAlias = str
Pattern: TypeAlias = str
PTY_Icon: TypeAlias = str
TTY_Icon: TypeAlias = str

# Used by cli.OptionsAsker:
OptName: TypeAlias = str
OptSpec: TypeAlias = dict[OptName, Any]


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
    def with_highlighting(cls,
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


class FormatterFieldSig(NamedTuple):
    '''
    '''

    lit: FormatterLiteral
    name: FormatterFname
    spec: FormatterFormatSpec
    conv: FormatterConversion

    @classmethod
    def from_str(cls, fmt: FormatStr) -> Self:
        '''
        Convert a format string field to a field signature.

        :param fmt: The format string to convert
        :type fmt: :class:`FormatStr`
        '''
        try:
            parsed = tuple(Formatter().parse(fmt))

        except ValueError:
            err = f"Invalid format string: {fmt!r}"
            raise BrokenFormatStringError(err) from None

        if not parsed:
            err = f"The format string {fmt!r} contains no fields."
            raise FormatStringError(err)

        field = parsed[0]

        # Does the field have a fieldname?
        if field[1] == '':
            # No; it's positional.
            start = len(field[0])
            err = (
                f"The format string field at character {start} in {fmt!r} is "
                f"missing a fieldname.\n"
                 "Positional fields ('{}' for example) are not allowed "
                 "for this operation."
            )
            raise MissingFieldnameError(err)

        sig = cls(*field)
        return sig

    def as_string(self,
        with_literal: bool = True,
        with_conv: bool = True,
    ) -> FormatStr:
        '''
        Recreate a format string field from a single field signature.

        :param with_literal: Include the signature's :class:`FormatterLiteral`,
            defaults to ``True``
        :type with_literal: :class:`bool`

        :param with_conv: Include the signature's :class:`FormatterConversion`,
            defaults to ``True``
        :type with_conv: :class:`bool`

        :returns: The format string represented by the signature
        :rtype: :class:`FormatStr`
        '''
        inside_braces = self.name
        if with_conv and self.conv is not None:
            inside_braces += '!' + self.conv
        inside_braces += ':' + self.spec if self.spec else self.spec
        fmt = '{' + inside_braces + '}'
        if with_literal:
            return self.lit + fmt
        return fmt


FmtStrStructure: TypeAlias = tuple[FormatterFieldSig]
'''
:class:`tuple[FormatterFieldSig]`

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
    fmt: FormatStr
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

    # The following field params are mutually exclusive with `fmt`.
    field_order: Required[list[FieldName]]
    field_definitions: dict[FieldName, FieldSpec]
    field_icons: dict[FieldName, Icon]
    separator: Separator
    separators: Sequence[PTY_Separator, TTY_Separator]

    # The `fmt` param is mutually exclusive with all field params.
    fmt: FormatStr


class BarTemplateSpec(BarSpec, total=False):
    '''A dict representation of :class:`mybar.templates.BarTemplate` constructor parameters.'''
    config_file: PathLike
    debug: bool


