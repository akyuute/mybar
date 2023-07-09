__all__ = (
    'FieldSpec',
    'BarSpec',
    'BarConfigSpec',
    'CMDLineSpec',
)


from os import PathLike

# from .field import FieldSpec
# from .bar import BarConfig, BarConfigSpec, BarSpec
from ._types import (
    Args,
    FieldName,
    FormatStr,
    Icon,
    Kwargs,
    PTY_Icon,
    PTY_Separator,
    Separator,
    TTY_Icon,
    TTY_Separator,
)

from collections.abc import Callable, Sequence
from typing import ParamSpec, Required, TypedDict, TypeVar


Bar = TypeVar('Bar')
P = ParamSpec('P')

class FieldSpec(TypedDict, total=False):
    '''
    Specifies the structure of :class:`mybar.Field` constructor
    parameters.
    '''
    name: Required[FieldName]
    func: Callable[P, str]
    icon: Icon
    template: FormatStr
    interval: float
    clock_align: bool
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
    '''
    Specifies the structure of :class:`mybar.Bar` constructor parameters.
    '''
    refresh: float
    run_once: bool
    count: int
    clock_align: bool
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
    '''
    Specifies the structure of :class:`mybar.bar.BarConfig` constructor
    parameters.
    '''
    config_file: PathLike
    debug: bool
    field_icons: dict[FieldName, Icon]


class CmdOptionSpec(TypedDict, total=False):
    '''
    Specifies the structure of command options.
    '''
    dump_config: bool | int = 4

