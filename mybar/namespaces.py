__all__ = (
    'FieldSpec',
    'BarSpec',
    'BarConfigSpec',
    'ConfigFileSpec',
    '_CmdOptionSpec',
)


from ast import Dict, List
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

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ParamSpec, Required, TypedDict, TypeVar


Bar = TypeVar('Bar')
P = ParamSpec('P')


class FieldConfigSpec(TypedDict, total=False):
    '''
    A dict specifying the structure of :class:`mybar.Field` constructor
    parameters as represented in config files and in the command line.
    '''
    name: FieldName
    icon: Icon | Sequence[ASCII_Icon, Unicode_Icon]
    template: FormatStr
    interval: float
    clock_align: bool
    overrides_refresh: bool
    threaded: bool
    always_show_icon: bool
    run_once: bool
    constant_output: str
    args: Args
    kwargs: Kwargs
    command: str
    script: PathLike
    allow_multiline: bool

class FieldSpec(FieldConfigSpec, total=False):
    '''
    A dict specifying the structure of :class:`mybar.Field` constructor
    parameters.
    '''
    func: Callable[P, str]
    setup: Callable[P, P.kwargs]
    bar: Bar

    unserializable = ('func', 'setup', 'bar')


class BarSpec(TypedDict, total=False):
    '''
    A dict specifying the structure of :class:`mybar.Bar` constructor
    parameters.
    '''
    break_lines: bool
    clock_align: bool
    count: int
    debug: bool
    join_empty_fields: bool
    override_cooldown: float
    refresh: float
    run_once: bool
    thread_cooldown: float
    unicode: bool

    '''
    The following field params are mutually exclusive with `template`:
    '''
    fields: Iterable[Field | FieldName]
    field_order: Sequence[FieldName]
    separator: Separator | Sequence[ASCII_Separator, Unicode_Separator]

    '''
    The `template` param is mutually exclusive with all field params:
    '''
    template: FormatStr


class BarConfigSpec(BarSpec, total=False):
    '''
    Specify options for :class:`mybar.bar.BarConfig` and config files.
    '''
    debug: bool
    field_definitions: Mapping[FieldName, FieldSpec]
    field_icons: Mapping[FieldName, Sequence[ASCII_Icon, Unicode_Icon] | Icon]
    field_order: Iterable[FieldName]
    from_icons: Mapping[FieldName, Icon]


class _CmdOptionSpec(TypedDict, total=False):
    '''
    Specify special command line optional arguments.
    '''
    config_file: PathLike
    dump_config: bool | int = 4

