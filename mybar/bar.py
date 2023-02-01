#TODO: NewType()!
#TODO: Bar.as_generator()!
#TODO: collections.defaultdict, dict.fromkeys!
#TODO: Finish Mocp line!


__all__ = (
    'Bar',
    'run',
    'from_cli',
)


import asyncio
import json
import os
import sys
import threading
import time
from copy import deepcopy
from string import Formatter

from . import CONFIG_FILE, DEBUG
from . import cli
from . import utils
from .errors import *
from .field import Field
from ._types import (
    BarSpec,
    BarTemplateSpec,
    Separator,
    PTY_Separator,
    TTY_Separator,
    Line,
    FieldName,
    Icon,
    PTY_Icon,
    TTY_Icon,
    ConsoleControlCode,
    FormatStr,
    Pattern,
    Args,
    Kwargs,
    JSONText,
)

from collections.abc import Iterable, Sequence
from typing import IO, NoReturn, Required, TypeAlias, TypedDict, TypeVar

B = TypeVar('B')
Bar = TypeVar('Bar')
T = TypeVar('T')
BarTemplate = TypeVar('BarTemplate')

# Unix terminal escape code (control sequence introducer):
CSI: ConsoleControlCode = '\033['
CLEAR_LINE: ConsoleControlCode = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR: ConsoleControlCode = '?25l'
UNHIDE_CURSOR: ConsoleControlCode = '?25h'


class BarTemplate(dict):
    '''
    Build and transport Bar configs between files, dicts and command line args.

    :param options: Optional :class:`BarTemplateSpec` parameters that override those of `defaults`
    :type options: :class:`BarTemplateSpec`

    :param defaults: Parameters to use by default,
        defaults to :attr:`Bar._default_params`
    :type defaults: :class:`dict`

    .. note:: `options` and `defaults` must be :class:`dict` instances of form :class:`BarTemplateSpec`

    '''

    def __init__(self,
        options: BarTemplateSpec = {},
        defaults: BarTemplateSpec = None,
    ) -> None:
        if defaults is None:
            self.defaults = Bar._default_params.copy()
        else:
            self.defaults = defaults.copy()
        self.options = options.copy()

        # self.bar_spec = self.defaults | self.options
        self.update(self.defaults | self.options)
        self.file = None
        debug = self.options.pop('debug', None) or DEBUG

    def __repr__(self) -> str:
        cls = type(self).__name__
        file = self.file
        maybe_file = f"{file=}, " if file else ""
        bar_spec = {**self}
        return f"<{cls} {maybe_file}{bar_spec=}>"

    @classmethod
    def from_file(cls: BarTemplate,
        file: PathLike = None,
        defaults: BarTemplateSpec = None,
        overrides: BarTemplateSpec = {}
    ) -> BarTemplate:
        '''
        Return a new :class:`BarTemplate` from a config file path.

        :param file: The filepath to the config file,
            defaults to ``'~/.mybar.json'``
        :type file: :class:`PathLike`

        :param defaults: The base :class:`BarTemplateSpec` dict whose
            params the new :class:`BarTemplate` will override,
            defaults to :attr:`Bar._default_params`
        :type defaults: :class:`BarTemplateSpec`

        :param overrides: Additional param overrides to the config file
        :type overrides: :class:`BarTemplateSpec`

        :returns: A new :class:`BarTemplate` instance
        :rtype: :class:`BarTemplate`
        :raises: :class:`AskWriteNewFile` to ask the user for write
            permission when the requested file path does not exist
        '''
        if defaults is None:
            defaults = Bar._default_params
        overrides = overrides.copy()

        file_given = True if file or 'config_file' in overrides else False
        if file is None:
            file = overrides.pop('config_file', CONFIG_FILE)

        file_spec = {}
        absolute = os.path.abspath(os.path.expanduser(file))
        if os.path.exists(absolute):
            file_spec, text = cls.read_file(absolute)
        elif file_given:
            raise AskWriteNewFile(absolute)
        else:
            cls.write_file(absolute, overrides, defaults)

        options = file_spec | overrides
        t = cls(options, defaults)
        t.file = absolute
        return t

    @classmethod
    def from_stdin(cls: BarTemplate,
        write_new_file_dft: bool = False
    ) -> BarTemplate:
        '''Return a new :class:`BarTemplate` using args from STDIN.
        Prompt the user before writing a new config file if one does
        not exist.

        :param write_new_file_dft: Write new files by default,
            defaults to ``False``
        :type write_new_file_dft: :class:`bool`

        :returns: A new :class:`BarTemplate`
        :rtype: :class:`BarTemplate`
        '''
        parser = cli.Parser()
        try:
            bar_options = parser.parse_args()

        except FatalError as e:
            parser.error(e.msg)

        except OSError as e:
            err = f"{parser.prog}: error: {e}"
            parser.quit(err)

        try:
            template = cls.from_file(overrides=bar_options)

        except AskWriteNewFile as e:
            file = e.requested_file
            errmsg = (
                f"{parser.prog}: error: \n"
                f"The config file at {file} does not exist."
            )
            question = "Would you like to make it now?"
            write_options = {'y': True, 'n': False}
            default = 'ny'[write_new_file_dft]
            handler = cli.OptionsAsker(write_options, default, question)

            print(errmsg)
            write_new_file = handler.ask()
            if write_new_file:
                cls.write_file(file, bar_options)
                print(f"Wrote new config file at {file}")
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
        :rtype: tuple[:class:`BarTemplateSpec`, :class:`JSONText`]
        '''
        with open(file, 'r') as f:
            data = json.load(f)
            text = f.read()
        return data, text

    @staticmethod
    def write_file(
        file: PathLike,
        obj: BarTemplateSpec = {},
        defaults: BarSpec = None
    ) -> None:
        '''Write :class:`BarTemplateSpec` params to a JSON file.

        :param file: The file to write to
        :type file: :class:`PathLike`
        :param obj: The :class:`BarTemplateSpec` to write
        :type obj: :class:`BarTemplateSpec`, optional
        :param defaults: Any default parameters that `obj` should override,
            defaults to :attr:`Bar._default_params`
        :type defaults: :class:`BarSpec`
        '''
        if defaults is None:
            defaults = Bar._default_params.copy()

        obj = obj.copy()
        obj.pop('config_file', None)
        obj = defaults | obj

        dft_fields = Field._default_fields.copy()

        for name, field in dft_fields.items():
            new = dft_fields[name] = field.copy()
            for param in ('name', 'func', 'setup'):
                try:
                    del new[param]
                except KeyError:
                    pass

        obj['field_definitions'] = dft_fields

        with open(os.path.expanduser(file), 'w') as f:
            json.dump(obj, f, indent=4, ) #separators=(',\n', ': '))


class Bar:
    '''
    The class used for creating highly customizable status bars.

    :param fields: An iterable of default field names or :class:`Field` instances, defaults to ``None``
    :type fields: :class:`Iterable[Field | str]`

    :param fmt: A curly-bracket format string with field names, defaults to ``None``
    :type fmt: :class:`FormatStr`

    :param separator: The field separator when `fields` is given, defaults to ``'|'``
    :type separator: :class:`PTY_Separator` | :class:`TTY_Separator`

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
    :type separators: tuple[:class:`PTY_Separator`, :class:`TTY_Separator`], optional

    :param stream: The bar's output stream, defaults to :attr:`sys.stdout`
    :type stream: :class:`IO`


    :raises: :class:`InvalidOutputStreamError` when `stream` does
        not implement the IO protocol.
    :raises: :class:`IncompatibleArgsError` when
        neither `fmt` nor `fields` are given
    :raises: :class:`IncompatibleArgsError` when `fmt`
        is ``None`` but no `separator` or `separators`
        are given
    :raises: :class:`TypeError` when `fields` is not iterable, or when `fmt` is not a string
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

    def __init__(self,
        fields: Iterable[Field|str] = None,
        *,
        fmt: str = None,
        separator: str = '|',
        refresh_rate: float = 1.0,
        stream: IO = sys.stdout,
        run_once: bool = False,
        align_to_seconds: bool = True,
        join_empty_fields: bool = False,
        override_cooldown: float = 1/60,
        thread_cooldown: float = 1/8,

        # Set this to use different seps for different output streams:
        separators: Sequence[PTY_Separator, TTY_Separator] = None,

        debug: bool = DEBUG,  # Not yet implemented!
    ) -> None:
        # Ensure the output stream has the required methods:
        io_methods = ('write', 'flush', 'isatty')
        if not all(hasattr(stream, a) for a in io_methods):
            io_method_calls = [a + '()' for a in io_methods]
            joined = utils.join_options(io_method_calls, final_sep=' and ')
            raise InvalidOutputStreamError(
                f"Output stream {stream!r} needs {joined} methods."
            )
        self._stream = stream

        # Ensure required parameters are defined if no fmt is given:
        if fmt is None:
            if fields is None:
                raise IncompatibleArgsError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required."
                )

            if not hasattr(fields, '__iter__'):
                raise TypeError("The 'fields' argument must be iterable.")

            if separator is None and separators is None:
                raise IncompatibleArgsError(
                    "A separator is required when 'fmt' is None."
                )

            # Gather a list of field names:
            field_order = [
                getattr(f, 'name') if isinstance(f, Field)
                else f
                for f in fields
            ]

        elif not isinstance(fmt, str):
            raise TypeError(
                f"Format string 'fmt' must be a string, not {type(fmt)}"
            )

        else:
            field_order = self.parse_fmt(fmt)
            if fields is None:
                fields = list(field_order)

        self._field_order = field_order
        self._fields = self.convert_fields(fields)
        self._buffers = {name: '' for name in self._fields}

        self.fmt = fmt

        if separators is None:
            separators = (separator, separator)
        self._separators = separators
        # self._separator = None

        # Whether empty fields are joined when fmt is None:
        # (True shows two separators together for every blank field.)
        self.join_empty_fields = join_empty_fields

        # Set how often in seconds the bar is normally printed.
        self.refresh_rate = refresh_rate

        # Whether the bar should exit after printing once:
        self.run_once = run_once

        # Whether the bar should start at the top of a clock second:
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

        # Calling run() sets this Event. Unsetting it stops all fields:
        self._can_run = threading.Event()

        # The bar's async event loop:
        self._loop = asyncio.new_event_loop()

    def __repr__(self) -> str:
        names = self._field_order
        fields = utils.join_options(names, final_sep='', quote=True, limit=3)
        cls = type(self).__name__
        return f"{cls}(fields=[{fields}])"

    @classmethod
    def from_dict(cls: Bar,
        dct: BarSpec,
        ignore_with: Pattern | tuple[Pattern] | None = '//'
    ) -> Bar:
        '''Make a :class:`Bar` using a dict of :class:`Bar` parameters.
        Ignore keys and list elements starting with `ignore_with`,
        which is ``'//'`` by default.
        If :param ignore_with: is ``None``, do not remove any values.

        :param dct: The :class:`dict` to convert
        :type dct: :class:`dict`
        :param ignore_with: A pattern to ignore, defaults to ``'//'``
        :type ignore_with: Pattern | tuple[Pattern] | None, optional
        :returns: A new :class:`Bar`
        :rtype: :class:`Bar`
        :raises: :class:`IncompatibleArgsError` when
            no :attr:`field_order` or :attr:`fmt` is defined
        :raises: :class:`DefaultFieldNotFoundError` when a field
            in :attr:`field_order` or :attr:`fmt` cannot be found
            in :attr:`Field._default_fields`
        :raises: :class:`UndefinedFieldError` when a field
            in :attr:`field_order` or :attr:`fmt` is not properly defined in :attr:`field_definitions`
        :raises: :class:`InvalidFieldSpecError` when
            a field definition is not of the form :class:`FieldSpec`

        .. note:: `dct` must match the form :class:`BarSpec`.

        '''
        if ignore_with is None:
            data = deepcopy(dct)
        else:
            data = utils.scrub_comments(dct, ignore_with)

        bar_params = cls._default_params | data
        field_defs = bar_params.pop('field_definitions', {})
        field_order = bar_params.pop('field_order', None)
        field_icons = bar_params.pop('field_icons', {})

        if (fmt := bar_params.get('fmt')) is None:
            if field_order is None:
                raise IncompatibleArgsError(
                    "A bar format string 'fmt' is required when field "
                    "order list 'field_order' is undefined."
                )
        else:
            field_order = cls.parse_fmt(fmt)

        fields = []
        expctd_name="the name of a default or properly defined `custom` Field"
        for name in field_order:
            field_params = field_defs.get(name)

            match field_params:
                case None:
                    # The field is strictly default:
                    try:
                        field = Field.from_default(name)
                    except DefaultFieldNotFoundError:
                        exc = utils.make_error_message(
                            DefaultFieldNotFoundError,
                            # doing_what="parsing 'field_order'",  # Only relevant in context file parsing
                            blame=f"{name!r}",
                            expected=expctd_name
                        )
                        raise exc from None

                case {'custom': True}:
                    # The field is custom, so it is only defined in
                    # 'field_definitions' and will not be found as a default.
                    custom_name = field_params.pop('name', name)
                    del field_params['custom']
                    field = Field(**field_params, name=custom_name)

                case {}:
                    # The field is a default overridden by the user.
                    # Has the user given custom icons from the command line?
                    if name in field_icons:
                        cust_icon = field_icons.pop(name)
                        field_params['icons'] = (cust_icon, cust_icon)

                    try:
                        field = Field.from_default(name, field_params)
                    except DefaultFieldNotFoundError:
                        exc = utils.make_error_message(
                            UndefinedFieldError,
                            # doing_what="parsing 'field_order'",
                            blame=f"{name!r}",
                            expected=expctd_name,
                            epilogue=(
                                f"(In config files, remember to set "
                                f"custom = true "
                                f"for custom field definitions.)"
                            ),
                        )
                        raise exc from None

                case _:
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

            fields.append(field)

        return cls(fields=fields, **bar_params)

    @classmethod
    def from_file(cls: Bar, file: os.PathLike) -> Bar:
        '''
        Generate a new :class:`Bar` by reading a config file.

        :param file: The config file to read
        :type file: :class:`os.PathLike`
        '''
        template = BarTemplate.from_file(file)
        return cls.from_dict(template)

    @property
    def clearline_char(self) -> str:
        '''A special character printed to TTY streams between refreshes.
        Its purpose is to clear characters left behind by longer lines.
        '''
        if self._stream is None:
            return None
        clearline = CLEAR_LINE if self._stream.isatty() else ''
        return clearline

    @property
    def fields(self) -> tuple[Field]:
        '''A tuple of the bar's Field objects.'''
        return tuple(self._fields.values())

    @property
    def in_a_tty(self) -> bool:
        '''True if the bar was run from a terminal, otherwise False.'''
        if self._stream is None:
            return False
        return self._stream.isatty()

    @property
    def separator(self) -> Separator:
        '''The field separator as determined by the output stream.
        It defaults to the TTY sep (self._separators[1]) if no stream is set.
        '''
        if self._stream is None:
            # Default to using the terminal separator:
            return self._separators[1]
        return self._separators[self._stream.isatty()]

    @staticmethod
    def parse_fmt(fmt: FormatStr) -> list[FieldName]:
        '''Return a list of field names that should act as a field order.

        :param fmt: A format string to parse
        :type fmt: :class:`str`
        :returns: A list of field names that were found
        :rtype: list[str]
        :raises: :class:`BrokenFormatStringError` when the format string
            is malformed or contains positional fields
        '''
        try:
            field_names = [
                name for m in Formatter().parse(fmt)
                if (name := m[1]) is not None
            ]
        except ValueError:
            err = f"Invalid bar format string: {fmt!r}"
            raise BrokenFormatStringError(err) from None
        if '' in field_names:
            raise BrokenFormatStringError(
                "Bar format strings cannot contain positional fields ('{}')."
            )
        return field_names

    def convert_fields(self,
        fields: Iterable[FieldName | Field]
    ) -> dict[FieldName, Field]:
        '''Convert a list of field names or :class:`Field` instances to
        corresponding default Fields and return a dict mapping field
        names to Fields.

        :param fields: The iterable of fields to convert
        :type fields: :class:`Iterable[str]`
        :returns: A dict mapping field names to :class:`Field` instances
        :rtype: dict[FieldName, Field] ####
        :rtype: :class:`dict`[:class:`str`, :class:`Field`]
        :raises: :class:`DefaultFieldNotFoundError` when a string
            element of `fields` is not the name of a default Field
        :raises: :class:`InvalidFieldError` when an element
            of `fields` is neither the name of a default Field
            nor an instance of :class:`Field`
        '''
        converted = {}
        for field in fields:
            match field:
                case str():
                    converted[field] = Field.from_default(
                        name=field,
                        params={'bar': self}
                    )
                case Field():
                    field._bar = self
                    converted[field.name] = field
                case _:
                    raise InvalidFieldError(f"Invalid field: {field}")
        return converted

    def run(self, *, stream: IO = None, once: bool = None) -> None:
        '''Run the bar in the specified output stream.
        Block until an exception is raised and exit smoothly.

        :param stream: The IO stream in which to run the bar
        :type stream: :class:`IO`
        :param once: Whether to print the bar only once, defaults to ``False``
        :type once: :class:`bool`
        :returns: ``None``

        .. note::
            This method cannot be called within an async event loop.
        '''
        if stream is not None:
            self._stream = stream
        if once is not None:
            self.run_once = once

        # Allow the bar to run repeatedly in the same interpreter.
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        try:
            self._can_run.set()
            self._loop.run_until_complete(self._startup(self.run_once))

        except KeyboardInterrupt:
            pass

        finally:
            self._shutdown()


    async def _startup(self, run_once: bool) -> None:
        '''Schedule field coroutines, threads and the line printer to be run
        in parallel.'''
        overriding = False
        gathered = []

        for field in self._fields.values():
            if field.overrides_refresh:
                overriding = True

            if field.threaded:
                await field.send_to_thread(run_once)

            else:
                gathered.append(field.run(run_once))

        if run_once:
            await asyncio.gather(*gathered)
            while self._threads:
                # Wait for threads to finish:
                await asyncio.sleep(self._thread_cooldown)
            # self._print_one_line(self._make_one_line())
            self._print_one_line()

        else:
            if overriding:
                gathered.append(self._handle_overrides())
            await asyncio.gather(
                *gathered,
                self._continuous_line_printer()
            )

    def _shutdown(self) -> None:
        '''Notify fields that the bar has stopped and join threads.'''
        self._can_run.clear()
        for field in self._fields.values():
            if field.threaded and field._thread is not None:
                field._thread.join()

        if self.in_a_tty:
            self._stream.write('\n')
            self._stream.write(CSI + UNHIDE_CURSOR)

    async def _continuous_line_printer(self, end: str = '\r') -> None:
        '''The bar's primary line-printing mechanism.
        Fields are responsible for sending data to the bar buffers.
        This only writes using the current buffer contents.'''
        # Again, local variables may save time:
        using_format_str = (self.fmt is not None)
        sep = self.separator
        running = self._can_run.is_set
        clock = time.monotonic

        if self.in_a_tty:
            beginning = self.clearline_char + end
            self._stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = self.clearline_char

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        self._stream.flush()

        # Print something right away just so that the bar is not empty:
        # self._print_one_line(self._make_one_line())
        self._print_one_line()

##        if using_format_str:
##            line = self.fmt.format_map(self._buffers)
##        else:
##            line = sep.join(
##                buf for field in self._field_order
##                    if (buf := self._buffers[field])
##                    or self.join_empty_fields
##            )
##        self._stream.write(beginning + line + end)
##        self._stream.flush()

        if self.align_to_seconds:
            # Begin every refresh at the start of a clock second:
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))

        start_time = clock()
        while running():
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
                    # clock() % self.refresh_rate
                    (clock() - start_time) % self.refresh_rate  # Preserve offset
                )
            )

    async def _handle_overrides(self, end: str = '\r') -> None:
        '''Prints a line when fields with overrides_refresh send new data.'''
        # Again, local variables may save time:
        bar = self._bar
        sep = self.separator
        using_format_str = (self.fmt is not None)
        running = self._can_run.is_set

        if self.in_a_tty:
            beginning = clearline_char + end
        else:
            beginning = clearline_char

        start_time = time.time()
        while running():

            try:
                # Wait until a field with overrides_refresh sends new
                # data to be printed:
                field, contents = await bar._override_queue.get()
                # if DEBUG:
                    # logger.debug(f"handler: {field} {time.time() - start_time}")

            except RuntimeError as exc:
                # asyncio raises RuntimeError if the event loop closes
                # while queue.get() is waiting for a value.
                # if DEBUG:
                    # logger.debug(exc.args[0])
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
            # if DEBUG:
                # logger.debug(f"handler: sleeping for {self._override_cooldown}")
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

    def _print_one_line(self,
        # line: str,
        stream: IO = None,
        end: str = '\r'
    ) -> None:
        '''Print a line to the buffer stream only once.
        This method is not meant to be called from within a loop.
        '''
        if stream is None:
            stream = self._stream

        if self.in_a_tty:
            beginning = self.clearline_char + end
            # stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = self.clearline_char

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        stream.flush()

        stream.write(beginning + self._make_one_line() + end)
        # stream.write(beginning + line + end)
        stream.flush()


def from_cli() -> Bar:
    '''
    Return a new :class:`Bar` using command line arguments.

    :returns: a new :class:`Bar` using command line arguments
    :rtype: :class:`Bar`
    '''
    template = BarTemplate.from_stdin()
    return Bar.from_dict(template)


def run(once: bool = False, file: os.PathLike = None) -> NoReturn | None:
    '''
    Generate a new :class:`Bar` from the default config file and run it in STDOUT.

    :param once: Print the bar only once, defaults to ``False``
    :type once: :class:`bool`

    :param file: The config file to source, defaults to :attr:`mybar.CONFIG_FILE`
    :type file: :class:`os.PathLike`
    '''
    template = BarTemplate.from_file(CONFIG_FILE)
    bar = Bar.from_dict(template)
    bar.run(once=once)

