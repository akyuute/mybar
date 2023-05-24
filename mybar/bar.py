#TODO: {'uptime1': {'dft': 'uptime', ...}, 'uptime2': {'dft': 'uptime', ...}}
#TODO: Bar(synchronous=True)!
#TODO: Bar.as_generator()!
#TODO: Finish Mocp line!


__all__ = (
    'Bar',
    'run',
)


import asyncio
import json
import os
import sys
import threading
import time
from copy import deepcopy
from string import Formatter

from .constants import CONFIG_FILE, DEBUG
from . import cli
from . import utils
from .errors import *
from .field import Field, FieldPrecursor
from .formatting import FormatterFieldSig
from ._types import (
    Args,
    BarSpec,
    BarTemplateSpec,
    ConsoleControlCode,
    FieldName,
    FieldOrder,
    FmtStrStructure,
    FormatStr,
    Icon,
    JSONText,
    Kwargs,
    Line,
    PTY_Icon,
    PTY_Separator,
    Pattern,
    Separator,
    TTY_Icon,
    TTY_Separator,
)

from collections.abc import Iterable, Iterator, Sequence
from os import PathLike
from typing import IO, NoReturn, Required, Self, TypeAlias, TypedDict, TypeVar

Bar = TypeVar('Bar')


# Unix terminal escape code (control sequence introducer):
CSI: ConsoleControlCode = '\033['
CLEAR_LINE: ConsoleControlCode = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR: ConsoleControlCode = '?25l'
UNHIDE_CURSOR: ConsoleControlCode = '?25h'


class BarTemplate(dict):
    '''
    Build and transport Bar configs between files, dicts and command line args.

    :param options: Optional :class:`_types.BarTemplateSpec` parameters
        that override those of `defaults`
    :type options: :class:`_types.BarTemplateSpec`

    :param defaults: Parameters to use by default,
        defaults to :attr:`Bar._default_params`
    :type defaults: :class:`dict`

    .. note:: `options` and `defaults` must be :class:`dict` instances
        of form :class:`_types.BarTemplateSpec`

    '''

    def __init__(
        self,
        options: BarTemplateSpec = {},
        defaults: BarTemplateSpec = None,
    ) -> None:
        if defaults is None:
            self.defaults = Bar._default_params.copy()
        else:
            self.defaults = defaults.copy()
        self.options = options.copy()

        # This is why we subclass dict:
        self.update(self.defaults | self.options)
        self.file = None
        debug = self.options.pop('debug', None) or DEBUG

    def __repr__(self) -> str:
        cls = type(self).__name__
        file = self.file
        maybe_file = f"{file=}, " if file else ""
        params = {**self}
        return f"<{cls} {maybe_file}{params}>"

    @classmethod
    def from_file(
        cls,
        file: PathLike = None,
        *,
        defaults: BarTemplateSpec = None,
        overrides: BarTemplateSpec = {},
    ) -> Self:
        '''
        Return a new :class:`BarTemplate` from a config file path.

        :param file: The filepath to the config file,
            defaults to ``'~/.mybar.json'``
        :type file: :class:`PathLike`

        :param defaults: The base :class:`_types.BarTemplateSpec` dict whose
            params the new :class:`BarTemplate` will override,
            defaults to :attr:`Bar._default_params`
        :type defaults: :class:`_types.BarTemplateSpec`

        :param overrides: Additional param overrides to the config file
        :type overrides: :class:`_types.BarTemplateSpec`

        :returns: A new :class:`BarTemplate` instance
        :rtype: :class:`BarTemplate`
        :raises: :exc:`OSError` for issues with accessing the file
        '''
        if defaults is None:
            defaults = Bar._default_params
        overrides = overrides.copy()

        if file is None:
            file = overrides.pop('config_file', CONFIG_FILE)

        file_spec = {}
        absolute = os.path.abspath(os.path.expanduser(file))
        try:
            file_spec, text = cls.read_file(absolute)
        except OSError as e:
            raise e.with_traceback(None)

        options = file_spec | overrides
        t = cls(options, defaults)
        t.file = absolute
        return t

    @classmethod
    def from_stdin(
        cls,
        write_new_file_dft: bool = True
    ) -> Self:
        '''Return a new :class:`BarTemplate` using args from STDIN.
        Prompt the user before writing a new config file if one does
        not exist.

        :param write_new_file_dft: Automatically write new files,
            defaults to ``False``
        :type write_new_file_dft: :class:`bool`

        :returns: A new :class:`BarTemplate`
        :rtype: :class:`BarTemplate`
        '''
        parser = cli.Parser()
        try:
            bar_options = parser.parse_args()
        except FatalError as e:
            parser.error(e.msg)  # Shows usage

##        except OSError as e:
##            parser.quit(e)

        try:
            template = cls.from_file(overrides=bar_options)
        except OSError as e:
            file = e.filename
            msg = (f"The config file at {file!r} does not exist.")
            question = "Would you like to make it now?"
            write_options = {'y': True, 'n': False}
            default = 'ny'[write_new_file_dft]
            handler = cli.OptionsAsker(write_options, default, question)

            print(msg)
            write_new_file = handler.ask()
            if write_new_file:
                cls.write_file(file, bar_options)
                print(f"Wrote new config file at {file!r}")
                template = cls.from_file(file)
            else:
                parser.quit()

        return template

    @staticmethod
    def read_file(file: PathLike) -> tuple[BarTemplateSpec, JSONText]:
        '''
        Read a given config file.
        Convert its JSON contents to a dict and return it along with the
        raw text of the file.

        :param file: The file to convert
        :type file: :class:`PathLike`

        :returns: The converted file and its raw text
        :rtype: tuple[:class:`_types.BarTemplateSpec`, :class:`_types.JSONText`]
        '''
        with open(file, 'r') as f:
            data = json.load(f)
            text = f.read()
        return data, text

    @staticmethod
    def write_file(
        file: PathLike,
        obj: BarTemplateSpec = {},
        *,
        defaults: BarSpec = None
    ) -> None:
        '''Write :class:`BarTemplate` params to a JSON file.

        :param file: The file to write to
        :type file: :class:`PathLike`

        :param obj: The :class:`_types.BarTemplateSpec` to write
        :type obj: :class:`_types.BarTemplateSpec`, optional

        :param defaults: Any default parameters that `obj` should override,
            defaults to :attr:`Bar._default_params`
        :type defaults: :class:`_types.BarSpec`
        '''
        if defaults is None:
            defaults = Bar._default_params.copy()

        obj = obj.copy()
        obj.pop('config_file', None)
        obj = defaults | obj

        fields = Field._default_fields.copy()

        # Clean the dict of irrelevant Field implementation details:
        for name, field in fields.items():
            new = fields[name] = field.copy()
            for param in ('name', 'func', 'setup'):
                try:
                    del new[param]
                except KeyError:
                    pass

        obj['field_definitions'] = fields

        with open(os.path.expanduser(file), 'w') as f:
            json.dump(obj, f, indent=4, ) #separators=(',\n', ': '))


class Bar:
    '''
    Create highly customizable status bars.

    :param fields: An iterable of default field names or :class:`Field` instances, defaults to ``None``
    :type fields: :class:`Iterable[Field | str]`

    :param fmt: A curly-brace format string with field names, defaults to ``None``
    :type fmt: :class:`_types.FormatStr`

    :param separator: The field separator when `fields` is given, defaults to ``'|'``
    :type separator: :class:`_types.PTY_Separator` | :class:`_types.TTY_Separator`

    :param run_once: Whether the bar should print once and return, defaults to ``False``
    :type run_once: :class:`bool`

    :param refresh_rate: How often in seconds the bar automatically redraws itself, defaults to ``1.0``
    :type refresh_rate: :class:`float`

    :param align_to_seconds: Whether to synchronize redraws at the start of each new second (updates to the clock are shown accurately), defaults to ``True``
    :type align_to_seconds: :class:`bool`

    :param join_empty_fields: Whether to draw separators around fields with no content, defaults to ``False``
    :type join_empty_fields: :class:`bool`

    :param override_cooldown: Cooldown in seconds between handling sequential field overrides, defaults to ``1/60``
    :type override_cooldown: :class:`float`

    :param thread_cooldown: How long a field thread loop sleeps after checking if the bar is still running,
        defaults to ``1/8``.
        Between executions, unlike async fields, a threaded field sleeps for several iterations of
        `thread_cooldown` seconds that always add up to :attr:`Field.interval` seconds.
        Between sleeps, it checks if the bar has stopped.
        A shorter cooldown means more chances to check if the bar has stopped and
        a faster exit time.
    :type thread_cooldown: :class:`float`

    :param separators: A tuple of 2 strings that separate fields when `fields` is given.
        Note: The `separator` parameter sets both of these automatically.
        The first string is used in graphical (PTY) environments where support for Unicode is more likely.
        The second string is used in terminal (TTY) environments where only ASCII is supported.
        This enables the same :class:`Bar` instance to use the most optimal separator automatically.
    :type separators: tuple[:class:`_types.PTY_Separator`, :class:`_types.TTY_Separator`], optional

    :param stream: The bar's output stream, defaults to :attr:`sys.stdout`
    :type stream: :class:`IO`


    :raises: :exc:`errors.InvalidOutputStreamError` when `stream` does
        not implement the IO protocol
    :raises: :exc:`errors.IncompatibleArgsError` when
        neither `fmt` nor `fields` are given
    :raises: :exc:`errors.IncompatibleArgsError` when
        `fmt` is ``None`` but no `separator` or `separators` are given
    :raises: :exc:`TypeError` when `fields` is not iterable, or when `fmt` is not a string
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
        'refresh_rate': 1.0,
        'field_order': list(_default_field_order)
    }

    def __init__(
        self,
        fmt: str = None,
        fields: Iterable[Field | str] = None,
        *,
        field_order: Iterable[FieldName] = None,
        separator: str = '|',
        refresh_rate: float = 1.0,
        stream: IO = sys.stdout,
        run_once: bool = False,
        align_to_seconds: bool = True,
        join_empty_fields: bool = False,
        override_cooldown: float = 1/8,
        thread_cooldown: float = 1/8,

        # Set this to use different seps for different output streams:
        separators: Sequence[PTY_Separator, TTY_Separator] = None,

        debug: bool = DEBUG,  # Not yet implemented!
    ) -> None:
        # Ensure the output stream has the required methods:
        self._check_stream(stream)
        self._stream = stream


        # Check all the required parameters.
        if fields is None:

            if fmt is None:
                raise IncompatibleArgsError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required."
                )

            elif not isinstance(fmt, str):
                raise TypeError(
                    f"Format string 'fmt' must be a string, not {type(fmt)}"
                )

            # Good to use fmt:
            else:
                #NOTE: `fields` here are :class:`FormatterFieldSig`
                # _normalize_fields() takes care of this later.
                field_order, fields = self.parse_fmt(fmt)

        # Fall back to using fields.
        elif not hasattr(fields, '__iter__'):
            raise TypeError("The 'fields' argument must be iterable.")

        elif separator is None and separators is None:
            raise IncompatibleArgsError(
                "A separator is required when 'fmt' is None."
            )

        names, fields = self._normalize_fields(fields)

        # Preserve custom field order, enabling duplicates:
        if fmt is None and field_order is None:
            field_order = names

        self._fields = fields
        self._field_order = field_order
        self._buffers = dict.fromkeys(self._fields, '')

        self.fmt = fmt

        if separators is None:
            separators = (separator, separator)
        self._separators = separators

        # Whether empty fields are joined when fmt is None:
        # (True shows two separators together for every blank field.)
        self.join_empty_fields = join_empty_fields

        # How often in seconds the bar is regularly printed:
        self.refresh_rate = refresh_rate

        # Whether the bar should exit after printing once:
        self.run_once = run_once

        # Whether the bar is printed at the top of every clock second:
        self.align_to_seconds = align_to_seconds

        # Make a queue to which fields with overrides_refresh send updates.
        # Process only one item per refresh cycle to reduce flickering in a PTY.
        self._override_queue = asyncio.Queue(maxsize=1)

        # How long should the queue checker sleep before processing a new item?
        # (A longer cooldown means less flickering):
        self._override_cooldown = override_cooldown

        # Staggering a thread's refresh interval across several sleeps with the
        # length of this cooldown prevents it from blocking for the entire
        # interval when the bar stops.
        # (A shorter cooldown means faster exits):
        self._thread_cooldown = thread_cooldown

        # The set of field threads the bar must wait to join before
        # printing a line with run_once:
        self._threads = set()

        self._printer_thread = None
        self._timely_fields = []

        # Calling run() sets this Event. Unsetting it stops all fields:
        self._can_run = threading.Event()
        self.running = self._can_run.is_set

        # The bar's async event loop:
        self._loop = asyncio.new_event_loop()

    def __contains__(self, other: FieldPrecursor) -> bool:
        if isinstance(other, str):
            weak_test = (other in self._field_order)
            return weak_test

        elif isinstance(other, Field):
            less_weak_test = (other.name in self._field_order)
            return less_weak_test

        # Shall we test for identical objects?
        else:
            # No, too specific for a dunder-method.
            return False

    def __eq__(self, other: Bar) -> bool:
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

        if self.fmt == other.fmt:
            return True
        return False

    def __iter__(self) -> Iterator:
        return iter(field for field in self._fields.values())

    def __len__(self) -> int:
        return len(self._field_order)

    def __repr__(self) -> str:
        names = self._field_order
        fields = utils.join_options(names, final_sep='', limit=3)
        cls = type(self).__name__
        return f"{cls}(fields=[{fields}])"

    @classmethod
    def from_template(
        cls,
        tmpl: BarTemplate,
        *,
        ignore_with: Pattern | tuple[Pattern] | None = '//'
    ) -> Self:
        '''
        Return a new :class:`Bar` from a :class:`BarTemplate`
        or a dict of :class:`Bar` parameters.
        Ignore keys and list elements starting with `ignore_with`,
        which is ``'//'`` by default.
        If :param ignore_with: is ``None``, do not remove any values.

        :param tmpl: The :class:`dict` to convert
        :type tmpl: :class:`dict`

        :param ignore_with: A pattern to ignore, defaults to ``'//'``
        :type ignore_with: :class:`_types.Pattern` | tuple[:class:`_types.Pattern`] | ``None``, optional

        :returns: A new :class:`Bar`
        :rtype: :class:`Bar`

        :raises: :exc:`errors.IncompatibleArgsError` when
            no `field_order` is defined
            or when no `fmt` is defined
        :raises: :exc:`errors.DefaultFieldNotFoundError` when either
            a field in `field_order` or
            a field in `fmt`
            cannot be found in :attr:`Field._default_fields`
        :raises: :exc:`errors.UndefinedFieldError` when
            a custom field name in `field_order` or
            a custom field name in `fmt`
            is not properly defined in `field_definitions`
        :raises: :exc:`errors.InvalidFieldSpecError` when
            a field definition is not of the form :class:`_types.FieldSpec`

        .. note:: `tmpl` can be a regular :class:`dict`
        as long as it matches the form of :class:`_types.BarSpec`.

        '''
        if ignore_with is None:
            data = deepcopy(tmpl)
        else:
            data = utils.scrub_comments(tmpl, ignore_with)

        bar_params = cls._default_params | data
        field_order = bar_params.pop('field_order', None)
        field_icons = bar_params.pop('field_icons', {})  # From the CLI
        field_defs = bar_params.pop('field_definitions', {})

        if (fmt := bar_params.get('fmt')) is None:
            if field_order is None:
                raise IncompatibleArgsError(
                    "A bar format string 'fmt' is required when field "
                    "order list 'field_order' is undefined."
                )
        else:
            field_order, field_sigs = cls.parse_fmt(fmt)

        # Ensure icon assignments correspond to valid fields:
        for name in field_icons:
            if name not in field_order:
                deduped = dict.fromkeys(field_order)  # Preserve order
                expctd_from_icons = utils.join_options(deduped)
                exc = utils.make_error_message(
                    InvalidFieldError,
                    doing_what="parsing custom Field icons",
                    blame=f"{name!r}",
                    expected=f"a Field name from among {expctd_from_icons}",
                    epilogue="Only assign icons to Fields that will be in the Bar."
                )
                raise exc from None

        # Gather Field parameters and instantiate them:
        fields = {}
        expctd_name="the name of a default or properly defined `custom` Field"

        for name in field_order:
            field_params = field_defs.get(name)

            match field_params:
                case None:
                    # The field is strictly default.
                    if name in fields:
                        continue

                    try:
                        field = Field.from_default(name)
                    except DefaultFieldNotFoundError:
                        exc = utils.make_error_message(
                            DefaultFieldNotFoundError,
                            # doing_what="parsing 'field_order'",  # Only relevant to config file parsing
                            blame=f"{name!r}",
                            expected=expctd_name
                        )
                        raise exc from None

                case {'custom': True}:
                    # The field is custom, so it is only defined in
                    # 'field_definitions' and will not be found as a default.
                    name = field_params.pop('name', name)
                    del field_params['custom']
                    field = Field(**field_params, name=name)

                case {}:
                    # The field is a default overridden by the user.
                    # Are there custom icons given from the CLI?
                    if name in field_icons:
                        cust_icon = field_icons.pop(name)
                        field_params['icons'] = (cust_icon, cust_icon)

                    try:
                        field = Field.from_default(name, overrides=field_params)
                    except DefaultFieldNotFoundError:
                        exc = utils.make_error_message(
                            UndefinedFieldError,
                            # doing_what="parsing 'field_order'",
                            blame=f"{name!r}",
                            expected=expctd_name,
                            epilogue=(
                                f"(In config files, remember to set "
                                f"`custom=true` "
                                f"for custom field definitions.)"
                            ),
                        )
                        raise exc from None

                case _:
                    # Edge cases.
                    exc = utils.make_error_message(
                        InvalidFieldSpecError,
                        # doing_what="parsing 'field_definitions'",
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
        return bar

    @classmethod
    def from_file(cls, file: PathLike = None) -> Self:
        '''
        Generate a new :class:`Bar` by reading a config file.

        :param file: The config file to read, defaults to :obj:`constants.CONFIG_FILE`
        :type file: :class:`PathLike`

        :returns: A new :class:`Bar`
        :rtype: :class:`Bar`
        '''
        template = BarTemplate.from_file(file)
        return cls.from_template(template)

    @classmethod
    def from_cli(cls) -> Self:
        '''
        Return a new :class:`Bar` using command line arguments.

        :returns: a new :class:`Bar` using command line arguments
        :rtype: :class:`Bar`
        '''
        template = BarTemplate.from_stdin()
        return cls.from_template(template)

##    @classmethod
##    def as_generator(cls, 

    @property
    def clearline_char(self) -> str:
        '''
        A special character printed to TTY streams between refreshes.
        Its purpose is to clear characters left behind by longer lines.
        '''
        if self._stream is None:
            return None
        clearline = CLEAR_LINE if self._stream.isatty() else ''
        return clearline

    @property
    def fields(self) -> tuple[Field]:
        '''A tuple of the bar's Field objects.'''
        return tuple(self)

    @property
    def in_a_tty(self) -> bool:
        '''True if the bar was run from a terminal, otherwise False.'''
        if self._stream is None:
            return False
        return self._stream.isatty()

    @property
    def separator(self) -> Separator:
        '''
        The field separator as determined by the output stream.
        It defaults to the TTY sep (self._separators[1]) if no stream is set.
        '''
        if self._stream is None:
            # Default to using the terminal separator:
            return self._separators[1]
        return self._separators[self._stream.isatty()]

    def append(self, field: FieldPrecursor) -> Self:
        '''
        Append a new Field to the bar.

        `field` will be converted to :class:`Field`,
        appended to the field list and joined to the end of the bar output.
        If :attr:`Bar.fmt` is defined, it will override the new field order.

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
        appended to the field list and joined to the end of the bar output.
        If :attr:`Bar.fmt` is defined, it will override the new field order.
        the fields are displayed.

        :param field: The fields to append
        :type field: :class:`FieldPrecursor`

        :returns: The modified bar
        :rtype: :class:`Bar`
        '''
        #TODO: Consider Bar.fmt += FormatterFieldSig(field).as_string()
        names, normalized = self._normalize_fields(fields)
        self._fields.update(normalized)
        self._buffers.update(dict.fromkeys(names, ''))
        self._field_order.extend(names)
        return self


    @staticmethod
    def parse_fmt(
        fmt: FormatStr
    ) -> tuple[FieldOrder, FmtStrStructure]:
        '''
        Return a dict mapping field names to :class:`FormatterFieldSig`.
        Keys of the dict can act as a field order.

        :param fmt: A format string to parse
        :type fmt: :class:`FormatStr`

        :returns: A dict of fields found in the format string
        :rtype: :class:`dict[FieldName, FormatterFieldSig]`

        :raises: :exc:`errors.BrokenFormatStringError` when `fmt`
            is malformed
        :raises: :exc:`errors.MissingFieldnameError` when `fmt`
            contains positional fields
        '''
        try:
            sigs = tuple(
                FormatterFieldSig(*field)
                for field in Formatter().parse(fmt)
            )

        except ValueError:
            err = f"Invalid bar format string: {fmt!r}"
            raise BrokenFormatStringError(err) from None

        if any(sig.name == '' for sig in sigs):
            # There's a positional field somewhere.
            # Issue an actionable error message:
            epilogue = (
                "Bar format string fields must all have fieldnames."
                "\nPositional fields ('{}' for example) are not allowed."
            )
            raise MissingFieldnameError.with_highlighting(sigs, epilogue)

        names = tuple(name for sig in sigs if (name := sig.name) is not None)

        return names, sigs

    @staticmethod
    def _check_stream(stream: IO) -> None | NoReturn:
        '''Raise TypeError if the output stream has the required methods.'''
        io_methods = ('write', 'flush', 'isatty')
        if not all(hasattr(stream, a) for a in io_methods):
            io_method_calls = [a + '()' for a in io_methods]
            joined = utils.join_options(io_method_calls, final_sep='and')
            raise TypeError(
                f"Output stream {stream!r} needs {joined} methods."
            )

    def _normalize_fields(
        self,
        fields: Iterable[FieldPrecursor],
    ) -> tuple[FieldOrder, dict[FieldName, Field]]:
        '''
        Convert :class:`Field` precursors to :class:`Field` instances.

        Take an iterable of field names, :class:`Field` instances, or
        :class:`FormatterFieldSig` tuples and convert them to
        corresponding default Fields.
        Return a dict mapping field names to Fields, along with a
        duplicate-preserving list of field names that were found.

        :param fields: An iterable of :class:`Field` primitives to convert
        :type fields: :class:`Iterable[FieldPrecursor]`

        :returns: A dict mapping field names to :class:`Field` instances
        :rtype: :class:`tuple[FieldOrder, dict[FieldName, Field]]`

        :raises: :exc:`errors.InvalidFieldError` when an element
            of `fields` is not a :class:`FieldPrecursor`
        '''
        normalized = {}
        names = []

        for field in fields:
            if isinstance(field, Field):
                new_field = field
                new_field._bar = self

            elif isinstance(field, str):
                new_field = Field.from_default(
                    name=field,
                    overrides={'bar': self},
                )

            elif isinstance(field, FormatterFieldSig):
                if field.name is None:
                    continue
                new_field = Field.from_default(
                    name=field.name,
                    overrides={'bar': self}
                )
                new_field._fmtsig = field.as_string()

            else:
                raise InvalidFieldError(f"Invalid field precursor: {field!r}")

            normalized[new_field.name] = new_field
            names.append(new_field.name)

        return names, normalized

    def line_generator(self):
        while self.running():
            if self._timely_fields:
                self._preload_timely_fields()
            line = self._make_one_line()
            yield line

    def run(self, once: bool = None, stream: IO = None) -> None:
        '''
        Run the bar in the specified output stream.
        Block until an exception is raised and exit smoothly.

        :param stream: The IO stream in which to run the bar,
            defaults to :attr:`Bar._stream`
        :type stream: :class:`IO`

        :param once: Whether to print the bar only once, defaults to ``False``
        :type once: :class:`bool`

        :returns: ``None``

        .. note::
            This method cannot be called within an async event loop.
        '''
        if stream is not None:
            self._stream = stream
        if once is None:
            once = self.run_once

        # Allow the bar to run repeatedly in the same interpreter.
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        try:
            self._can_run.set()
            self._loop.run_until_complete(self._startup(once))

            # Keep the main thread running for fields handled by the
            # printer thread.
            if self._timely_fields and not once:
               while self.running():
                    time.sleep(self._thread_cooldown)

        except KeyboardInterrupt:
            pass

        finally:
            self._shutdown()

    async def _startup(self, run_once: bool = False) -> None:
        '''
        Run field coroutines and threads in parallel.
        '''
        overriding = False
        gathered = []

        for field in self:
            if field.overrides_refresh:
                overriding = True

            if field.threaded:
                field.send_to_thread(run_once=run_once)
            elif field.timely:
                self._timely_fields.append(field)
            else:
                gathered.append(field.run(run_once))

        if run_once:
            await asyncio.gather(*gathered)
            # Wait for threads to finish:
            while self._threads:
                await asyncio.sleep(self._thread_cooldown)
            self._print_one_line()

        else:
##            if overriding:  # Currently disabled!
##                gathered.append(self._handle_overrides())
            self._start_printer()
            await asyncio.gather(*gathered)

    def _preload_timely_fields(self) -> None:
        for f in self._timely_fields:
            if asyncio.iscoroutinefunction(fun := f._func):
                result = asyncio.run(fun(*f.args, **f.kwargs))
            else:
                result = fun(*f.args, **f.kwargs)
            self._buffers[f.name] = result

    def _start_printer(self, end: str = '\r') -> None:
        self._printer_thread = threading.Thread(
            target=self._threaded_continuous_line_printer,
            name='PRINTER',
            args=(end,)
        )
        self._threads.add(self._printer_thread)
        self._printer_thread.start()

    def _threaded_continuous_line_printer(self, end: str = '\r') -> None:
        '''
        The bar's primary line-printing mechanism.
        Fields with the ``timely`` attribute get run here.
        Other fields are responsible for sending data to bar buffers.
        This only writes using the current buffer contents.

        :param end: The string appended to the end of each line
        :type end: :class:`str`
        '''
        using_format_str = (self.fmt is not None)
        sep = self.separator
        clock = time.monotonic

        if self.in_a_tty:
            beginning = self.clearline_char + end
            self._stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = self.clearline_char

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        self._stream.flush()

        # Print something right away just so that the bar is not empty:
        self._print_one_line()

        if self.align_to_seconds:
            # Begin every refresh at the start of a clock second:
            clock = time.time
            time.sleep(1 - (clock() % 1))

        step = self._thread_cooldown
        count = 0
        needed = round(self.refresh_rate / step)

        start_time = clock()
        while self.running():

            # Sleep until the next refresh cycle, pausing for a bit in
            # case the bar stops.
            if count < needed:
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
                    )
                )
                continue

            count = 0

            # Run time-sensitive fields right away:
            for f in self._timely_fields:
                if asyncio.iscoroutinefunction(fun := f._func):
                    result = asyncio.run(fun(*f.args, **f.kwargs))
                else:
                    result = fun(*f.args, **f.kwargs)
                self._buffers[f.name] = result

            if using_format_str:
                line = self.fmt.format_map(self._buffers)
            else:
                line = sep.join(
                    buf for field in self._field_order
                        if (buf := self._buffers[field])
                        or self.join_empty_fields
                )

            self._stream.write(beginning + line + end)
            self._stream.flush()

            # Sleep only until the next possible refresh to keep the
            # refresh cycle length consistent and prevent drifting.
            time.sleep(
                # Time until next refresh:
                self.refresh_rate - (
                    # Get the current latency, which can vary:
                    (clock() - start_time) % self.refresh_rate  # Preserve offset
                )
            )

    def _shutdown(self) -> None:
        '''
        Notify fields that the bar has stopped and join threads.
        Also print a newline and unhide the cursor if the bar was
        running in a TTY.
        '''
        self._can_run.clear()
        for thread in self._threads:
            thread.join()
        ##for field in self:
        ##    if field.threaded and field._thread is not None:
        ##        field._thread.join()

        if self.in_a_tty:
            self._stream.write('\n')
            self._stream.write(CSI + UNHIDE_CURSOR)

    async def _continuous_line_printer(self, end: str = '\r') -> None:
        '''
        The bar's primary line-printing mechanism.
        Fields are responsible for sending data to the bar buffers.
        This only writes using the current buffer contents.

        :param end: The string appended to the end of each line
        :type end: :class:`str`
        '''
        using_format_str = (self.fmt is not None)
        sep = self.separator
        clock = time.monotonic

        if self.in_a_tty:
            beginning = self.clearline_char + end
            self._stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = self.clearline_char

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        self._stream.flush()

        # Print something right away just so that the bar is not empty:
        self._print_one_line()

        if self.align_to_seconds:
            # Begin every refresh at the start of a clock second:
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))

        start_time = clock()
        while self.running():
            if using_format_str:
                line = self.fmt.format_map(self._buffers)
            else:
                line = sep.join(
                    buf for field in self._field_order
                        if (buf := self._buffers[field])
                        or self.join_empty_fields
                )

            self._stream.write(beginning + line + end)
            self._stream.flush()

            # Sleep only until the next possible refresh to keep the
            # refresh cycle length consistent and prevent drifting.
            await asyncio.sleep(
                # Time until next refresh:
                self.refresh_rate - (
                    # Get the current latency, which can vary:
                    (clock() - start_time) % self.refresh_rate  # Preserve offset
                )
            )

    async def _handle_overrides(self, end: str = '\r') -> None:
        '''
        Print a line when fields with overrides_refresh send new data.

        :param end: The string appended to the end of each line
        :type end: :class:`str`
        '''
        sep = self.separator
        using_format_str = (self.fmt is not None)

        if self.in_a_tty:
            beginning = self.clearline_char + end
        else:
            beginning = self.clearline_char

        start_time = time.time()
        while self.running():

            try:
                # Wait until a field with overrides_refresh sends new
                # data to be printed:
                field, contents = await self._override_queue.get()
                # if DEBUG:
                    # logger.debug(f"handler: {field} {time.time() - start_time}")

            except RuntimeError:
                # asyncio raises RuntimeError if the event loop closes
                # while queue.get() is waiting for a value.
                return

            if using_format_str:
                line = self.fmt.format_map(self._buffers)
            else:
                line = sep.join(
                    buf for field in self._field_order
                        if (buf := self._buffers[field])
                        or self.join_empty_fields
                )

            self._stream.write(beginning + line + end)
            self._stream.flush()
            await asyncio.sleep(self._override_cooldown)

    def _make_one_line(self) -> Line:
        '''Make a line using the bar's field buffers.
        This method is not meant to be called from within a loop.
        '''
        if self.fmt is not None:
            line = self.fmt.format_map(self._buffers)
        else:
            line = self.separator.join(
                buf for field in self._field_order
                    if (buf := self._buffers[field])
                    or self.join_empty_fields
            )
        return line

    def _print_one_line(self, end: str = '\r') -> None:
        '''Print a line to the buffer stream only once.
        This method is not meant to be called from within a loop.

        :param end: The string appended to the end of each line
        :type end: :class:`str`
        '''
        if self.in_a_tty:
            beginning = self.clearline_char + end
        else:
            beginning = self.clearline_char

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        self._stream.flush()

        if self._timely_fields:
            self._preload_timely_fields()
        self._stream.write(beginning + self._make_one_line() + end)

        self._stream.flush()


def run(once: bool = False, file: PathLike = None) -> NoReturn | None:
    '''
    Generate a new :class:`Bar` and run it in STDOUT.
    Optionally generate the bar from a config file.

    :param once: Print the bar only once, defaults to ``False``
    :type once: :class:`bool`

    :param file: The config file to source, defaults to :obj:`constants.CONFIG_FILE`
    :type file: :class:`PathLike`
    '''
    bar = Bar.from_file(file)
    bar.run(once=once)

