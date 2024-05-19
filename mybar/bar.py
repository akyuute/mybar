#TODO: Implement dynamic icons!
#TODO: Finish Mocp line!


__all__ = ('Bar', 'BarConfig')


import asyncio
import json
import os
import sys
import threading
import time
from copy import deepcopy

import scuff

from .constants import (
    CLEAR_LINE,
    CONFIG_FILE,
    CSI,
    DEBUG,
    HIDE_CURSOR,
    UNHIDE_CURSOR,
)

from . import cli
from . import field_funcs
from . import utils
from .errors import *
from .field import Field, FieldPrecursor
from .formatting import FmtStrStructure, FormatterFieldSig
from .namespaces import BarConfigSpec, BarSpec, FieldSpec, _CmdOptionSpec
from ._types import (
    Args,
    ASCII_Icon,
    ASCII_Separator,
    Bar,
    BarConfig,
    ConsoleControlCode,
    FieldName,
    FieldOrder,
    FileContents,
    FormatStr,
    Icon,
    JSONText,
    Kwargs,
    Line,
    Pattern,
    Separator,
    Unicode_Icon,
    Unicode_Separator,
)

from collections.abc import Iterable, Iterator, Mapping, Sequence
from os import PathLike
from typing import (
    IO,
    NoReturn,
    Required,
    Self,
)


class BarConfig(dict):
    '''
    Build and transport configs between files, dicts and command lines.

    :param options: Optional :class:`namespaces.BarConfigSpec` parameters
        that override those of `defaults`
    :type options: :class:`namespaces.BarConfigSpec`

    :param defaults: Parameters to use by default,
        defaults to :attr:`Bar._default_params`
    :type defaults: :class:`Mapping`

    .. note:: `options` and `defaults` must be :class:`Mapping` instances
        of form :class:`namespaces.BarConfigSpec`

    '''
    def __init__(
        self,
        options: BarConfigSpec = {},
        defaults: BarConfigSpec = None,
    ) -> None:
        if defaults is None:
            self.defaults = Bar._default_params.copy()
        else:
            self.defaults = defaults.copy()
        self.options = options.copy()

        # This is why we subclass dict:
        self.update(self.defaults | self.options)
        self.file = None
        self.file_contents = None
        debug = self.options.pop('debug', None) or DEBUG

    def __repr__(self) -> str:
        cls = type(self).__name__
        file = self.file
        maybe_file = f"{file=}, " if file else ""
        params = self.copy()
        return f"<{cls} {maybe_file}{params}>"

    @classmethod
    def from_file(
        cls,
        file: PathLike = None,
        *,
        defaults: BarConfigSpec = None,
        overrides: BarConfigSpec = {},
    ) -> BarConfig:
        '''
        Return a new :class:`BarConfig` from a config file path.

        :param file: The filepath to the config file,
            defaults to ``'~/.mybar.json'``
        :type file: :class:`PathLike`

        :param defaults: The base :class:`namespaces.BarConfigSpec` dict
            whose params the new :class:`BarConfig` will override,
            defaults to :attr:`Bar._default_params`
        :type defaults: :class:`namespaces.BarConfigSpec`

        :param overrides: Additional param overrides to the config file
        :type overrides: :class:`namespaces.BarConfigSpec`

        :returns: A new :class:`BarConfig` instance
        :rtype: :class:`BarConfig`
        :raises: :exc:`OSError` for issues with accessing the file
        '''
        if defaults is None:
            defaults = Bar._default_params
        overrides = overrides.copy()

        if file is None:
            file = overrides.pop('config_file', CONFIG_FILE)
        absolute = os.path.abspath(os.path.expanduser(file))
        try:
            from_file, text = cls._read_file(absolute)
        except OSError as e:
            raise e.with_traceback(None)

        options = utils.nested_update(from_file, overrides)

        # Remove invalid API-only parameters:
        options = cls._remove_unserializable(options)

        config = cls(options, defaults)
        config.file = absolute
        config.file_contents = text
        return config

    @classmethod
    def _unify_field_defs(cls, params: BarConfigSpec) -> BarConfigSpec:
        '''
        Gather loose Field definitions and join them with any others
        found in a `field_definitions` item of `params`, if it exists.
        Return a config with a unified `field_definitions`.

        :param params: The config to process
        :type params: :class:`namespaces.BarConfigSpec`

        :returns: The processed config
        :rtype: :class:`namespaces.BarConfigSpec`
        '''
        all_except_defs = (
            BarConfigSpec.__optional_keys__
            | BarConfigSpec.__required_keys__
        )
        config = params.copy()
        defs = config.pop('field_definitions', {})
        for maybe_def in params:
            if maybe_def not in all_except_defs:
                defs[maybe_def] = config.pop(maybe_def)

        # Remove command line options:
        for option in _CmdOptionSpec.__optional_keys__:
            defs.pop(option, None)

        config['field_definitions'] = defs
        return config

    @classmethod
    def write_with_approval(
        cls,
        file: PathLike = None,
        overrides: BarSpec = {},
    ) -> bool:
        '''
        Write a config file to `file`.
        Prompt the user for approval before writing.

        :param file: Write to this file, defaults to :obj:`CONFIG_FILE`
        :type file: :class:`PathLike`

        :returns: ``True`` if the file was written, ``False`` if not
        '''
        if file is None:
            file = CONFIG_FILE

        approved = cli.FileManager._get_write_new_approval(
            file,
            dft_choice=True
        )
        if approved:
            try:
                cls._write_file(file, overrides)
            except FileNotFoundError as e:
                try:
                    path = os.path.dirname(e.filename)
                    os.mkdir(path)
                    print(f"Made new directory {path!r}")
                    cls._write_file(file, overrides)
                except OSError as e:
                    raise e from None

            print(f"Wrote new config file to {file!r}")
            return True

        return False

    @classmethod
    def _read_file(cls, file: PathLike) -> tuple[BarSpec, FileContents]:
        '''
        Read a given config file.
        Parse and convert its contents to a dict.
        Return the dict along with the raw text of the file.

        :param file: The file to convert
        :type file: :class:`PathLike`

        :returns: The converted file and its raw text contents
        :rtype: tuple[
            :class:`namespaces.BarSpec`,
            :class:`FileContents`
            ]
        '''
        absolute = os.path.abspath(os.path.expanduser(file))
        p = scuff.FileParser(file=absolute)
        from_file = p.to_py()
        data = cls._unify_field_defs(from_file)
        text = p._string
        return data, text

    @classmethod
    def _write_file(
        cls,
        file: PathLike,
        spec: BarSpec = {},
        indent: int = 4,
        *,
        defaults: BarSpec = None
    ) -> None:
        '''
        Write :class:`BarConfig` params to a config file.

        :param file: The file to write to
        :type file: :class:`PathLike`

        :param spec: The :class:`namespaces.BarSpec` to write
        :type spec: :class:`namespaces.BarSpec`, optional

        :param indent: How many spaces to indent by, defaults to ``4``
        :type indent: :class:`int`

        :param defaults: Any default parameters that `spec` should
            override, defaults to :attr:`Bar._default_params`
        :type defaults: :class:`namespaces.BarSpec`
        '''
        if defaults is None:
            defaults = Bar._default_params.copy()
        un_pythoned = cls._remove_unserializable(defaults | spec)
        absolute = os.path.abspath(os.path.expanduser(file))
        if absolute == CONFIG_FILE and not os.path.exists(absolute):
            cli.FileManager._maybe_make_config_dir()

        text = scuff.py_to_scuff(un_pythoned)
        with open(absolute, 'w') as f:
            f.write(text)

    @staticmethod
    def _read_json(file: PathLike) -> tuple[BarConfigSpec, JSONText]:
        '''
        Read a given JSON config file.
        Convert its contents to a dict and return it along with the
        raw text of the file.

        :param file: The file to convert
        :type file: :class:`PathLike`

        :returns: The converted file and its raw text
        :rtype: tuple[:class:`namespaces.BarConfigSpec`,
            :class:`JSONText`]
        '''
        absolute = os.path.abspath(os.path.expanduser(file))
        with open(absolute, 'r') as f:
            data = json.load(f)
            text = f.read()
        return data, text

    @classmethod
    def _write_json(
        cls,
        file: PathLike,
        spec: BarSpec = {},
        indent: int = 4,
        *,
        defaults: BarSpec = None
    ) -> None:
        '''
        Write :class:`BarConfig` params to a JSON file.

        :param file: The file to write to
        :type file: :class:`PathLike`

        :param spec: The :class:`namespaces.BarSpec` to write
        :type spec: :class:`namespaces.BarSpec`, optional

        :param indent: How many spaces to indent by, defaults to ``4``
        :type indent: :class:`int`

        :param defaults: Any default params that `spec` should override,
            defaults to :attr:`Bar._default_params`
        :type defaults: :class:`namespaces.BarSpec`
        '''
        if defaults is None:
            defaults = Bar._default_params.copy()
        un_pythoned = cls._remove_unserializable(defaults | spec)
        absolute = os.path.abspath(os.path.expanduser(file))
        if absolute == CONFIG_FILE and not os.path.exists(absolute):
            cli.FileManager._maybe_make_config_dir()
        with open(absolute, 'w') as f:
            json.dump(un_pythoned, f, indent=indent, ) #separators=(',\n', ': '))

    @staticmethod
    def _remove_unserializable(spec: BarConfigSpec) -> BarConfigSpec:
        '''
        Remove Python-specific API elements like functions which cannot
        be serialized for writing to config files.

        :param spec: The :class:`namespaces.BarConfigSpec` to convert
        :type spec: :class:`namespaces.BarConfigSpec`
        '''
        newspec = spec.copy()
        newspec.pop('config_file', None)

        fields = newspec.pop('field_definitions', {})
        # Throw away unneeded Field implementation details:
        newfields = {}
        for name, params in fields.items():
            newfields[name] = {
                param: val for param, val in params.items()
                if param not in FieldSpec.unserializable
            }
        if newfields:
            newspec['field_definitions'] = newfields

        return newspec

    #TODO: _as_scuff() method!
    @classmethod
    def _as_json(cls, spec: BarSpec = {}, indent: int = 4) -> JSONText:
        '''
        Return a :class:`BarConfig` as a string with JSON formatting.

        :param spec: The :class:`namespaces.BarSpec` to convert
        :type spec: :class:`namespaces.BarSpec`, optional

        :param indent: How many spaces to indent by, defaults to ``4``
        :type indent: :class:`int`
        '''
        cleaned = cls._remove_unserializable(spec)
        return json.dumps(cleaned, indent=indent)


class Bar:
    '''
    Create highly customizable status bars.

    :param fields: An iterable of default field names or :class:`Field`
        instances, defaults to ``None``
    :type fields: Iterable[:class:`FieldPrecursor`]

    :param template: A curly-brace format string with field names,
        defaults to ``None``
    :type template: :class:`FormatStr`

    :param separator: The field separator string when `fields` is given,
        or a sequence of 2 of these, defaults to ``'|'``
        The first string is used in terminal environments where
        only ASCII is supported.
        The second string is used in graphical environments where
        support for Unicode is more likely.
        This enables the same :class:`Bar` instance to use the most
        optimal separator automatically.
    :type separator: :class:`Separator` |
        Sequence[:class:`ASCII_Separator`,
        :class:`Unicode_Separator`], optional

    :param refresh: How often in seconds the bar automatically redraws
        itself, defaults to ``1.0``
    :type refresh: :class:`float`

    :param count: Only print the bar this many times, or never stop
        when ``None``, defaults to ``None``
    :type count: :class:`int` | ``None``

    :param break_lines: Print each refresh after a newline character
        (``'\\n'``), defaults to False
    :type break_lines: :class:`bool`

    :param clock_align: Whether to synchronize redraws at the start of
        each new second (updates to the clock are shown accurately),
        defaults to ``True``
    :type clock_align: :class:`bool`

    :param join_empty_fields: Whether to draw separators around fields
        with no content, defaults to ``False``
    :type join_empty_fields: :class:`bool`

    :param override_cooldown: Cooldown in seconds between handling
        sequential field overrides, defaults to ``1/60``.
        A longer cooldown typically means less flickering.
    :type override_cooldown: :class:`float`

    :param thread_cooldown: How long a field thread loop sleeps after
        checking if the bar is still running, defaults to ``1/8``.
        Between executions, unlike async fields, a threaded field sleeps
        for several iterations of `thread_cooldown` seconds that always
        add up to :attr:`Field.interval` seconds.
        Between sleeps, it checks if the bar has stopped.
        A shorter cooldown means more chances to check if the bar has
        stopped and a faster exit time.
    :type thread_cooldown: :class:`float`

    :param unicode: Use Unicode variants of `separator` and Field
        icons, if given; optional
    :type unicode: :class:`bool`

    :param stream: The bar's output stream,
        defaults to :attr:`sys.stdout`
    :type stream: :class:`IO`

    :raises: :exc:`errors.InvalidOutputStreamError` when `stream` does
        not implement the IO protocol
    :raises: :exc:`errors.IncompatibleArgsError` when
        neither `template` nor `fields` are given
    :raises: :exc:`errors.IncompatibleArgsError` when
        `template` is ``None``
        but no `separator` given
    :raises: :exc:`TypeError` when `fields` is not iterable, or when
        `template` is not a string
    '''

    _default_field_order = [
        'hostname',
        'uptime',
        'cpu_usage',
        'cpu_temp',
        'mem_usage',
        'disk_usage',
        'battery',
        'net_stats',
        'datetime',
    ]

    _default_params = {
        'separator': '|',
        'refresh': 1.0,
        'field_order': list(_default_field_order)
    }

    def __init__(
        self,
        template: FormatStr = None,
        fields: Iterable[FieldPrecursor] = None,
        *,
        field_order: Iterable[FieldName] = None,
        separator: Separator | Sequence[ASCII_Separator,
                                        Unicode_Separator] = '|',
        refresh: float = 1.0,
        count: int | None = None,
        break_lines: bool = False,
        clock_align: bool = True,
        join_empty_fields: bool = False,
        override_cooldown: float = 1/8,
        thread_cooldown: float = 1/8,
        unicode: bool = True,
        stream: IO = sys.stdout,
        debug: bool = DEBUG,  # Not yet implemented!
    ) -> None:
        # Ensure the output stream has the required methods:
        self._check_stream(stream)
        self._stream = stream

        # Check all the required parameters.
        if fields is None:

            if template is None:
                raise IncompatibleArgsError(
                    f"Either a list of Fields `fields` "
                    f"or a format string `template` is required."
                )

            elif not isinstance(template, str):
                raise TypeError(
                    f"Format string `template` must be a string, "
                    f"not {type(template)}"
                )

            # Good to use template:
            else:
                parsed = FmtStrStructure.from_str(template)
                parsed.validate_fields(Field._default_fields, True, True)
                # names = parsed.get_names()
                fields = parsed

        # Fall back to using fields.
        elif not isinstance(separator, Iterable):
            raise TypeError("The `fields` argument must be iterable.")

        if unicode is None:
            unicode = not self.stream.isatty()
        self._unicode = unicode

        if isinstance(separator, str):
            separator = (separator, separator)
        elif not isinstance(separator, Sequence):
            raise TypeError("`separator` must be a sequence")
        self._separators = separator

        field_names, fields = self._normalize_fields(fields)

        # Preserve custom field order, enabling duplicates:
        if field_order is None:
            field_order = field_names
        self._fields = fields
        self._field_order = field_order
        self._buffers = dict.fromkeys(self._fields, '')

        self.template = template

        self._endline = '\n' if break_lines is True else '\r'
        self.join_empty_fields = join_empty_fields
        self.refresh_rate = refresh

        self.count = count
        self._print_countdown = count

        self.clock_align = clock_align
        self._override_queue = asyncio.Queue(maxsize=1)
        self._override_cooldown = override_cooldown
        self._thread_cooldown = thread_cooldown

        # The dict mapping Field names to the threads running them.
        # The bar must join each before printing a line with ``count=1``:
        self._threads = {}
        self._printer_thread = None
        self._printer_loop = None
        self._funcs = {}

        self._coros = {}
        self._timely_fields = []

        # Calling run() sets this Event. Unsetting it stops all fields:
        self._can_run = threading.Event()
        self._running = self._can_run.is_set

        # The bar's async event loop:
        # self._thread_loop = asyncio.new_event_loop()
        self._loop = asyncio.new_event_loop()

        self._file = None
        self._config = None

    def __contains__(self, field: FieldPrecursor) -> bool:
        if isinstance(other, str):
            weak_test = (field in self._field_order)
            return weak_test
        elif isinstance(field, Field):
            less_weak_test = (field.name in self._field_order)
            return less_weak_test
        else:
            return False

    def __eq__(self, other: Self) -> bool:
        if not all(
            getattr(self, attr) == getattr(other, attr)
            for attr in (
                '_fields',
                '_field_order',
                '_separators',
                'join_empty_fields',
            )
        ):
            return False
        if self.template == other.template:
            return True
        return False

    def __iter__(self) -> Iterator:
        # Include duplicates.
        return iter(self._fields[name] for name in self._field_order)

    def __len__(self) -> int:
        # Include duplicates.
        return len(self._field_order)

    def __repr__(self) -> str:
        names = self._field_order
        fields = utils.join_options(names, final_sep='', limit=3)
        cls = type(self).__name__
        return f"{cls}(fields=[{fields}])"

    @property
    def config(self) -> BarConfig:
        '''
        The :class:`BarConfig` used to instantiate the Bar, if applicable.
        '''
        return self._config

    @property
    def file(self) -> PathLike:
        '''
        The config file used to instantiate the Bar, if applicable.
        '''
        return self._file

    @classmethod
    def from_config(
        cls,
        config: BarConfig,
        *,
        overrides: BarSpec = {},
        ignore_with: Pattern | tuple[Pattern] | None = '//'
    ) -> Self:
        '''
        Return a new :class:`Bar` from a :class:`BarConfig`
        or a dict of :class:`Bar` parameters.
        Ignore keys and list elements starting with `ignore_with`,
        which is ``'//'`` by default.
        If :param ignore_with: is ``None``, do not remove any values.

        :param config: The :class:`Mapping` to convert
        :type config: :class:`Mapping`

        :param overrides: Replace items in `config` with these params,
            defaults to ``{}``
        :type overrides: :class:`namespaces.BarSpec`

        :param ignore_with: A pattern to ignore, defaults to ``'//'``
        :type ignore_with: :class:`Pattern` |
            tuple[:class:`Pattern`] | ``None``, optional

        :returns: A new :class:`Bar`
        :rtype: :class:`Bar`

        :raises: :exc:`errors.IncompatibleArgsError` when
            no `field_order` is defined
            or when no `template` is defined
        :raises: :exc:`errors.DefaultFieldNotFoundError` when either
            a field in `field_order` or
            a field in `template`
            cannot be found in :attr:`Field._default_fields`
        :raises: :exc:`errors.UndefinedFieldError` when
            a custom field name in `field_order` or
            a custom field name in `template`
            is not properly defined in `field_definitions`
        :raises: :exc:`errors.InvalidFieldSpecError` when
            a field definition is not of the form
            :class:`namespaces.FieldSpec`

        .. note:: `config` can be a regular :class:`Mapping`
            as long as it matches the form of :class:`namespaces.BarSpec`.

        '''
        if ignore_with is None:
            data = deepcopy(config)
        else:
            data = utils.scrub_comments(config, ignore_with)
        data = utils.nested_update(data, overrides)
        bar_params = utils.nested_update(cls._default_params, data)

        field_order = bar_params.pop('field_order', None)
        field_icons = bar_params.pop('field_icons', {})
        field_defs = bar_params.pop('field_definitions', {})
        template = bar_params.get('template')
        if field_icons is None:
            field_icons = {}

        if template is None:
            if not field_order:
                # field_order = list(field_icons)
                raise IncompatibleArgsError(
                    "A bar format string `template` is required"
                    " when field order list `field_order` is undefined"
                    " or empty."
                )
        else:
            parsed = FmtStrStructure.from_str(template)
            parsed.validate_fields(Field._default_fields, True, True)
            field_order = parsed.get_names()

        # Gather Field parameters and instantiate new Fields:
        fields = {}
        expctd_name = (
            "the name of a default or"
            " properly defined `custom` Field in field list"
        )

        for name in field_order:
            field_params = field_defs.get(name)

            cust_icons = field_icons.get(name, None)
            if cust_icons is not None:
                if isinstance(cust_icons, str):
                    # When one icon is given, override both defaults:
                    cust_icons = (cust_icons, cust_icons)

            match field_params:
                case None:
                    # The field is strictly default.
                    if name in fields:
                        continue

                    if cust_icons is not None:
                        overrides = {'icon': cust_icons}
                    else:
                        overrides = {}

                    try:
                        field = Field.from_default(name, overrides=overrides)
                    except DefaultFieldNotFoundError:
                        exc = utils.make_error_message(
                            DefaultFieldNotFoundError,
                            blame=f"{name!r}",
                            expected=expctd_name
                        )
                        raise exc from None

                case {'custom': True}:
                    # The field is custom, so it is only defined in
                    # 'field_definitions' and will not be found as a default.
                    name = field_params.pop('name', name)
                    del field_params['custom']
                    field_params['icon'] = cust_icons
                    field = Field(**field_params, name=name)

                case {}:
                    # The field is a default overridden by the user.
                    if cust_icons is not None:
                        field_params['icon'] = cust_icons
                    try:
                        field = Field.from_default(
                            name,
                            overrides=field_params
                        )
                    except DefaultFieldNotFoundError:
                        exc = utils.make_error_message(
                            UndefinedFieldError,
                            doing_what=f"parsing {name!r} definition",
                            blame=f"{name!r}",
                            expected=expctd_name,
                            epilogue=(
                                f"(In config files, remember to set"
                                f" `custom=true`"
                                f" for custom field definitions.)"
                            ),
                        )
                        raise exc from None

                case _:
                    # Edge cases.
                    exc = utils.make_error_message(
                        InvalidFieldSpecError,
                        doing_what=f"parsing {name!r} definition",
                        details=(
                            f"Invalid Field specification: {field_params!r}",
                        ),
                        indent_level=1
                    )
                    raise exc from None

            fields[name] = field

        bar = cls(
            fields=tuple(fields.values()),
            field_order=field_order,
            **bar_params
        )
        bar._config = config
        if hasattr(config, 'file'):
            bar._file = config.file
        return bar

    @classmethod
    def from_file(
        cls,
        file: PathLike = None,
        overrides: BarSpec = {}
    ) -> Self:
        '''
        Generate a new :class:`Bar` by reading a config file.

        :param file: The config file to read,
            defaults to :obj:`constants.CONFIG_FILE`
        :type file: :class:`PathLike`

        :param overrides: Replace items in `config` with these params,
            defaults to ``{}``
        :type overrides: :class:`namespaces.BarSpec`

        :returns: A new :class:`Bar`
        :rtype: :class:`Bar`
        '''
        config = BarConfig.from_file(file)
        bar = cls.from_config(config, overrides=overrides)
        bar._file = config.file
        return bar

    @property
    def clearline_char(self) -> str:
        '''
        A special character printed to TTY streams between refreshes.
        Its purpose is to clear characters left behind by longer lines.
        '''
        if self._stream is None:
            return None
        if self._stream.isatty() and self._endline == '\r':
            return CLEAR_LINE + self._endline
        else:
            return ''
        return clearline

    @property
    def fields(self) -> tuple[Field]:
        '''A tuple of the bar's Field objects.'''
        return tuple(self)

    @property
    def in_a_tty(self) -> bool:
        '''
        ``True`` if the bar was run from a terminal, otherwise ``False``.
        '''
        if self._stream is None:
            return False
        return self._stream.isatty()

    @property
    def running(self) -> bool:
        '''
        ``True`` if the bar is currently running, otherwise ``False``.
        '''
        return self._running()

    @property
    def separator(self) -> Separator:
        '''
        The field separator as determined by the output stream.
        Defaults to the TTY sep (self._separators[0]) if no stream is set.
        '''
        if self._stream is None:
            # Default to using the ASCII separator:
            return self._separators[0]
        return self._separators[self._unicode]

    def append(self, field: FieldPrecursor) -> Self:
        '''
        Append a new Field to the bar.

        `field` will be converted to :class:`Field`,
        appended to the field list and joined to the end of the bar.
        If :attr:`Bar.template` is defined,
        it will override the new field order.

        :param field: The field to append
        :type field: :class:`FieldPrecursor`

        :returns: The modified bar
        :rtype: :class:`Bar`
        '''
        (name,), normalized = self._normalize_fields((field,))
        self._fields.update(normalized)
        self._buffers[name] = ''
        self._field_order.append(name)
        return self

    def extend(self, fields: Iterable[FieldPrecursor]) -> Self:
        '''
        Append several new Fields to the bar.
        Field precursors in `fields` will be converted to :class:`Field`,
        appended to the field list and joined to the end of the bar.
        If :attr:`Bar.template` is defined,
        it will override the new field order.
        the fields are displayed.

        :param fields: The fields to append
        :type fields: Iterable[:class:`FieldPrecursor`]

        :returns: The modified bar
        :rtype: :class:`Bar`
        '''
        #TODO: Consider Bar.template += FormatterFieldSig(field).as_string()
        names, normalized = self._normalize_fields(fields)
        self._fields.update(normalized)
        self._buffers.update(dict.fromkeys(names, ''))
        self._field_order.extend(names)
        return self

    @staticmethod
    def _check_stream(
        stream: IO,
        raise_on_fail: bool = True
    ) -> NoReturn | None:
        '''
        Check if an IO stream has proper methods for use by :class:`Bar`.
        If any methods are missing,
            return ``False`` if `raise_on_fail` is ``False``, or
            raise :exc:`InvalidBarError` if `raise_on_fail` is ``True``.
        Otherwise, return ``True``.

        :param stream: The IO stream object to test
        :type stream: :class:`IO`

        :param raise_on_fail: Raise exception if `stream` fails the check,
            defaults to ``True``
        :type raise_on_fail: :class:`bool`

        :raises: :exc:`InvalidOutputStreamError` if `raise_on_fail` is
            ``True`` and the stream fails the test and lacks required
            instance methods
        '''
        io_methods = ('write', 'flush', 'isatty')
        if not all(hasattr(stream, m) for m in io_methods):
            if not raise_on_fail:
                return False
            io_method_calls = [m + '()' for m in io_methods]
            joined = utils.join_options(io_method_calls, final_sep='and')
            raise InvalidOutputStreamError(
                f"Output stream {stream!r} needs {joined} methods."
            ) from None
        return True

    def _normalize_fields(
        self,
        fields: Iterable[FieldPrecursor],
    ) -> tuple[FieldOrder, dict[FieldName, Field]]:
        '''
        Convert :class:`Field` precursors to :class:`Field` instances.

        Take an iterable of field names, :class:`Field` instances, or
        :class:`formatting.FormatterFieldSig` tuples and convert them to
        corresponding default Fields.
        Return a dict mapping field names to Fields, along with a
        duplicate-preserving list of field names that were found.

        :param fields: An iterable of :class:`FieldPrecursor` to convert
        :type fields: :class:`Iterable[FieldPrecursor]`

        :returns: The field order and a dict mapping field names to
            :class:`Field` instances
        :rtype: :class:`tuple[FieldOrder, Mapping[FieldName, Field]]`

        :raises: :exc:`errors.InvalidFieldError` when an element
            of `fields` is not a :class:`FieldPrecursor`
        '''
        normalized = {}
        names = []

        for field in fields:
            match field:
                case Field():
                    new_field = field
                    new_field._bar = self
                case str(name):
                    new_field = Field.from_default(
                        name=field,
                        overrides={'bar': self},
                    )
                case FormatterFieldSig(name=None):
                        # Skip sigs that lack fields:
                        continue
                case FormatterFieldSig():
                    new_field = Field.from_default(
                        name=field.name,
                        overrides={'bar': self}
                    )
                    new_field._fmtsig = field.as_string()
                case _:
                    msg = f"Invalid field precursor: {field!r}"
                    raise InvalidFieldError(msg)

            normalized[new_field.name] = new_field
            names.append(new_field.name)

        return names, normalized

    def run(
        self,
        once: bool = None,
        stream: IO = None,
        *,
        bg: bool = False
    ) -> None:
        '''
        Run the bar in the specified output stream.
        Block until an exception is raised and exit smoothly.

        :param stream: The IO stream in which to run the bar,
            defaults to :attr:`Bar._stream`
        :type stream: :class:`IO`

        :param once: Run the bar only once,
            defaults to internal state
        :type once: :class:`bool`

        :param bg: (Not fully implemented)
            Run the bar in the background without printing,
            defaults to ``False``
            This is used for :meth:`Bar.line_generator` and for testing.
        :type bg: :class:`bool`

        :returns: ``None``

        .. note::
            This method cannot be called within an async event loop.
        '''
        if self.count == 0:
            return

        try:
            if stream is not None:
                self._stream = stream
            if once is None:
                once = False if self.count is None else (self.count <= 1)
            elif once:
                self.count = 1

            # Allow the bar to run repeatedly in the same interpreter:
            if self._loop.is_closed():
                self._loop = asyncio.new_event_loop()

            self._prepare_fields()
            self._can_run.set()
            for thread in tuple(self._threads.values()):
                thread.start()
            self._preload_timely_fields()

            if not once:
                self._start_printer()

            # This blocks:
            self._loop.run_until_complete(self._gather_coros())

            if once:
                # Wait for threads to finish to get a full line, then print.
                while self._threads:
                    time.sleep(self._thread_cooldown)
                self._print_one_line()
                self._can_run.clear()

            # Prevent the bar from exiting when there are no coroutines:
            while self._running():
                time.sleep(self._thread_cooldown)

        except KeyboardInterrupt:
            pass

        finally:
            self._shutdown()

    def _prepare_fields(self) -> None:
        once = False if self.count is None else (self.count <= 1)
        overriding = False
        for field in self:
            if field.overrides_refresh:
                overriding = True

            if field.threaded:
                thread = field.make_thread(bar=self, run_once=once)
                self._threads[thread.name] = thread
            elif field.timely:
                self._timely_fields.append(field)
            else:
                self._coros[field.name] = field.run(bar=self, once=once)

##        if overriding:  # Currently disabled!
##            self._coros.append(self._handle_overrides())

    async def _gather_coros(self) -> None:
        '''
        Run :class:`Bar` coroutines in parallel.
        '''
        await asyncio.gather(*self._coros.values())

    def _preload_timely_fields(self) -> None:
        '''
        Update the bar buffers for Fields with :attr:`Field.timely`.
        This is used most often right before printing a line.
        '''
        for f in self._timely_fields:
            if f.is_async:
                result = self._loop.run_until_complete(
                    f._func(*f.args, **f.kwargs)
                )
            else:
                result = f._func(*f.args, **f.kwargs)

            if f.template is not None:
                contents = f.template.format(result, icon=f.icon)
            else:
                if f.always_show_icon or result:
                    contents = f.icon + result
                else:
                    contents = result
            self._buffers[f.name] = contents

    def _start_printer(self) -> None:
        '''
        Make a thread in which to run
            :meth:`Bar._threaded_continuous_line_printer` and start it.
        '''
        thread_name = 'PRINTER'
        self._printer_thread = threading.Thread(
            target=self._threaded_continuous_line_printer,
            name=thread_name,
        )
        self._threads[thread_name] = self._printer_thread
        self._printer_thread.start()

    def _threaded_continuous_line_printer(self) -> None:
        '''
        The bar's primary line-printing mechanism.
        Fields with the ``timely`` attribute get run here.
        They run once every refresh,
        effectively ignoring their ``interval`` attributes.
        Other fields are responsible for sending data to bar buffers.
        This only writes using the current buffer contents.
        '''
        # Make an event loop for this thread:
        self._printer_loop = asyncio.new_event_loop()

        # Flushing the buffer before writing to it fixes poor i3bar
        # alignment.
        self._stream.flush()

        beginning = self.clearline_char
        if self.in_a_tty:
            self._stream.write(CSI + HIDE_CURSOR)

        # Print something right away just so that the bar is not empty,
        # but not if the print count matters:
        if self.count is None:
            line = self._make_one_line()
            self._stream.write(beginning + line + self._endline)
            self._stream.flush()

        using_format_str = (self.template is not None)
        sep = self.separator
        clock = time.monotonic
        step = self._thread_cooldown
        needed = round(self.refresh_rate / step)
        count = 0
        first_cycle = True

        if self.clock_align:
            # Sleep until the beginning of the next second.
            clock = time.time
            time.sleep(1 - (clock() % 1))

        start_time = clock()
        while self._running():

            # Sleep until the next refresh cycle in little steps to
            # check if the bar has stopped.
            if count < needed and not first_cycle:
                count += 1

                # Latency from nonzero execution times causes drift,
                # where sleeps become out-of-sync and the bar skips
                # field updates.
                # This is especially noticeable in fields that update
                # routinely, such as the time.

                # To negate drift, when sleeping until the beginning of
                # the next cycle, we must also compensate for latency.
                time.sleep(
                    step - (
                        # Get the current latency, which can vary:
                        clock() % step
                        # (clock() - start_time) % step Preserves offset
                    )
                )
                continue

            count = 0
            if first_cycle:
                first_cycle = False

            # Run time-sensitive fields right away:
            for f in self._timely_fields:
                if f.is_async:
                    result = self._printer_loop.run_until_complete(
                        f._func(*f.args, **f.kwargs)
                    )
                else:
                    result = f._func(*f.args, **f.kwargs)

                if f.template is not None:
                    contents = f.template.format(result, icon=f.icon)
                else:
                    if f.always_show_icon or result:
                        contents = f.icon + result
                    else:
                        contents = result
                self._buffers[f.name] = contents

            if using_format_str:
                line = self.template.format_map(self._buffers)
            else:
                line = sep.join(
                    buf for field in self._field_order
                        if (buf := self._buffers[field])
                        or self.join_empty_fields
                )

            self._stream.write(beginning + line + self._endline)
            self._stream.flush()

            if self.count:
                self._print_countdown -= 1
                if self._print_countdown == 0:
                    self._can_run.clear()
                    self._printer_loop.stop()
                    self._printer_loop.close()
                    return

        self._printer_loop.stop()
        self._printer_loop.close()

    def _shutdown(self) -> None:
        '''
        Notify fields that the bar has stopped and join threads.
        Also print a newline and unhide the cursor if the bar was
        running in a terminal.
        '''
        self._can_run.clear()

        threads = tuple(self._threads.items())
        for thread_name, thread in threads:
            thread.join()
            self._threads.pop(thread_name)

        if self.in_a_tty:
            self._stream.write('\n')
            self._stream.write(CSI + UNHIDE_CURSOR)

    async def _handle_overrides(self) -> None:
        '''
        Print a line when fields with overrides_refresh send new data.
        '''
        sep = self.separator
        using_format_str = (self.template is not None)
        beginning = self.clearline_char

        start_time = time.time()
        while self._running():

            try:
                # Wait until a field with overrides_refresh sends new
                # data to be printed:
                field, contents = await self._override_queue.get()

            except RuntimeError:
                # asyncio raises RuntimeError if the event loop closes
                # while queue.get() is waiting for a value.
                return

            if using_format_str:
                line = self.template.format_map(self._buffers)
            else:
                line = sep.join(
                    buf for field in self._field_order
                        if (buf := self._buffers[field])
                        or self.join_empty_fields
                )

            self._stream.write(beginning + line + self._endline)
            self._stream.flush()
            await asyncio.sleep(self._override_cooldown)

    def _make_one_line(self) -> Line:
        '''Make a line using the bar's field buffers.
        This method is not meant to be called from within a loop.
        '''
        if self.template is not None:
            line = self.template.format_map(self._buffers)
        else:
            line = self.separator.join(
                buf for field in self._field_order
                    if (buf := self._buffers[field])
                    or self.join_empty_fields
            )
        return line

    def _print_one_line(self) -> None:
        '''Print a line to the buffer stream only once.
        This method is not meant to be called from within a loop.
        '''
        beginning = self.clearline_char

        if self._timely_fields:
            self._preload_timely_fields()
        self._stream.write(beginning + self._make_one_line() + self._endline)

        self._stream.flush()

    def line_generator(self) -> Iterator:
        '''
        Return a generator yielding status lines.
        This requires running the Fields in a separate thread,
        which has yet to be implemented.
        '''
        while self._running():
            if self._timely_fields:
                self._preload_timely_fields()
            line = self._make_one_line()
            yield line

    def current_line(self) -> Line:
        '''
        Return the most recently printed line.
        '''
        return self._make_one_line()

