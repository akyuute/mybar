__all__ = (
    'FieldSpec',
    'BarSpec',
    'BarConfigSpec',
    'CMDLineSpec',
)


from os import PathLike

from ._types import (
    Args,
    ASCII_Icon,
    ASCII_Separator,
    Field,
    FieldName,
    FormatStr,
    Icon,
    Kwargs,
    Separator,
    Unicode_Icon,
    Unicode_Separator,
)

from collections.abc import Callable, Iterable, Sequence
from typing import ParamSpec, Required, Sequence, TypedDict, TypeVar


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
    setup: Callable[P, P.kwargs]
    # Set this to use different icons for different output streams:
    icons: Sequence[[ASCII_Icon, Unicode_Icon]]


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
    fields: Iterable[Field | FieldName]
    field_order: Required[list[FieldName]]
    separator: Separator
    separators: Sequence[[ASCII_Separator, Unicode_Separator]]

    # The `template` param is mutually exclusive with all field params.
    template: FormatStr


class BarConfigSpec(BarSpec, total=False):
    '''
    Specifies the structure of :class:`mybar.bar.BarConfig` constructor
    parameters.
    '''
    debug: bool
    field_definitions: dict[FieldName, FieldSpec]
    field_icons: dict[FieldName, Icon]
    unicode: bool


class CmdOptionSpec(TypedDict, total=False):
    '''
    Specifies command options.
    '''
    config_file: PathLike
    dump_config: bool | int = 4

