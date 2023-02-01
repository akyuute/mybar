__all__ = (
    'BarSpec',
    'BarTemplateSpec',
    'Contents'
    'Separator',
    'PTY_Separator',
    'TTY_Separator',
    'Line',
    'FieldName',
    'Icon',
    'PTY_Icon',
    'TTY_Icon',
    'ConsoleControlCode',
    'FormatStr',
    'Pattern',
    'Args',
    'Kwargs',
    'JSONText',
)


from collections.abc import Callable, Sequence
from enum import Enum
from typing import ParamSpec, Required, TypeAlias, TypedDict, TypeVar
from os import PathLike


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

Args: TypeAlias = list
Kwargs: TypeAlias = dict

Bar = TypeVar('Bar')
P = ParamSpec('P')


class ColorEscaping(Enum):
    ANSI = 'ANSI'
    POLYBAR = 'POLYBAR'


class FieldSpec(TypedDict, total=False):
    '''A dict representation of Field constructor parameters.'''
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
    '''A dict representation of Bar constructor parameters.'''
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

    # The `fmt` params is mutually exclusive with all field params.
    fmt: FormatStr


class BarTemplateSpec(BarSpec, total=False):
    '''A dict representation of bar.Template constructor parameters.'''
    config_file: PathLike
    debug: bool


