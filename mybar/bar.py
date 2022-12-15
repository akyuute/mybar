#TODO: NewType()! TypedDict()!
#TODO: collections.defaultdict, dict.fromkeys!
#TODO: Finish Mocp line!
#TODO: Implement dynamic icons!


__all__ = (
    'Bar',
    'Template',
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

from mybar import CONFIG_FILE, DEBUG
from mybar.errors import *
from mybar.field import Field
from mybar.utils import (
    join_options,
    make_error_message,
    scrub_comments,
)


### Typing ###
from collections.abc import Iterable, Sequence
from typing import IO, NoReturn, TypeAlias

BarParamSpec: TypeAlias = dict[str]
PTY_Separator: TypeAlias = str
TTY_Separator: TypeAlias = str

FieldName: TypeAlias = str
FieldParamSpec: TypeAlias = dict[str]
Icon: TypeAlias = str
PTY_Icon: TypeAlias = str
TTY_Icon: TypeAlias = str

ConsoleControlCode: TypeAlias = str
FormatStr: TypeAlias = str

Args: TypeAlias = list
Kwargs: TypeAlias = dict

TemplateSpec: TypeAlias = dict[str]
JSONText: TypeAlias = str


# Unix terminal escape code (control sequence introducer):
CSI: ConsoleControlCode = '\033['
CLEAR_LINE: ConsoleControlCode = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR: ConsoleControlCode = '?25l'
UNHIDE_CURSOR: ConsoleControlCode = '?25h'


class Bar:

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
    ) -> None:
        # Ensure the output stream has the required methods:
        io_methods = ('write', 'flush', 'isatty')
        if not all(hasattr(stream, a) for a in io_methods):
            io_method_calls = [a + '()' for a in io_methods]
            joined = join_options(io_method_calls, final_sep=' and ')
            raise InvalidOutputStreamError(
                f"Output stream {stream!r} needs {joined} methods."
            )
        self._stream = stream

        # Ensure required parameters are defined if no fmt is given:
        if fmt is None:
            if fields is None:
                raise ValueError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required."
                )

            if not hasattr(fields, '__iter__'):
                raise ValueError("The 'fields' argument must be iterable.")

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
        fields = join_options(names, final_sep='', quote=True, limit=3)
        cls = type(self).__name__
        return f"{cls}(fields=[{fields}])"

    @classmethod
    def from_dict(cls,
        dct: BarParamSpec,
        ignore_with: str | tuple[str] | None = '//'
    ):
        '''Accept a mapping of Bar parameters.
        Ignore keys and list elements starting with '//' by default.
        '''
        if ignore_with is None:
            data = deepcopy(dct)
        else:
            data = scrub_comments(dct, ignore_with)

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
        for name in field_order:
            field_params = field_defs.get(name)

            match field_params:
                case None:
                    # The field is strictly default:
                    field = Field.from_default(name)
                    if field is None:
                        exc = make_error_message(
                            UndefinedFieldError,
                            doing_what="parsing 'field_order'",
                            blame=f"{name!r}",
                            expected=(
                                f"the name of a default Field or a "
                                f"custom field defined in 'field_definitions'"
                            ),
                            indent_level=1
                        )
                        raise exc

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

                    field = Field.from_default(name, field_params)

                    if field is None:
                        exc = make_error_message(
                            DefaultFieldNotFoundError,
                            doing_what="parsing 'field_definitions'",
                            blame=f"{name!r}",
                            expected="the name of a default Field",
                            epilogue=(
                                f"(In config files, remember to set "
                                f"custom = true "
                                f"for custom field definitions.)"
                            ),
                            indent_level=1
                        )
                        raise exc

                case _:

                    exc = make_error_message(
                        InvalidFieldSpecError,
                        doing_what="parsing 'field_definitions'",
                        details=(
                            f"Invalid Field specification: {field_params!r}",
                        ),
                        indent_level=1
                    )
                    raise exc

            fields.append(field)

        return cls(fields=fields, **bar_params)

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
    def separator(self) -> str:
        '''The field separator as determined by the output stream.
        It defaults to the TTY sep (self._separators[1]) if no stream is set.
        '''
        if self._stream is None:
            # Default to using the terminal separator:
            return self._separators[1]
        return self._separators[self._stream.isatty()]

    @staticmethod
    def parse_fmt(fmt: FormatStr) -> list[str]:
        '''Returns a list of field names that should act as a field order.'''
        try:
            field_names = [
                name
                for m in Formatter().parse(fmt)
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

    def convert_fields(self, fields: Iterable[str]) -> dict[str, Field]:
        '''Converts strings in a list of fields to corresponding default Fields
        and returns a dict mapping field names to Fields.'''
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
        '''Run the bar.
        Block until an exception is raised and exit smoothly.'''
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
                await asyncio.sleep(self._thread_cooldown)
            self._print_one_line(self._make_one_line())

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
        if using_format_str:
            line = self.fmt.format_map(self._buffers)
        else:
            line = self.separator.join(
                buf
                for field in self._field_order
                    if (buf := self._buffers[field])
                    or self.join_empty_fields
            )
        self._stream.write(beginning + line + end)
        self._stream.flush()

        if self.align_to_seconds:
            # Begin every refresh at the start of a clock second:
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))

        start_time = clock()
        while running():
            if using_format_str:
                line = self.fmt.format_map(self._buffers)
            else:
                line = self.separator.join(
                    buf
                    for field in self._field_order
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
                line = self.separator.join(
                    buf
                    for field in self._field_order
                        if (buf := self._buffers[field])
                        or self.join_empty_fields
                )

            self._stream.write(beginning + line + end)
            self._stream.flush()
            # if DEBUG:
                # logger.debug(f"handler: sleeping for {self._override_cooldown}")
            await asyncio.sleep(self._override_cooldown)

    def _make_one_line(self) -> str:
        '''Make a line using the bar's field buffers.
        This method is not meant to be called from within a loop.
        '''
        if self.fmt is not None:
            line = self.fmt.format_map(self._buffers)
        else:
            line = self.separator.join(
                buf
                for field in self._field_order
                    if (buf := self._buffers[field])
                    or self.join_empty_fields
            )
        return line

    def _print_one_line(self,
        line: str,
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
            stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = self.clearline_char

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        stream.flush()

        stream.write(beginning + line + end)
        stream.flush()


class Template:
    def __init__(self,
        options: TemplateSpec = {},
        defaults: TemplateSpec = None,
    ) -> None:
        if defaults is None:
            self.defaults = Bar._default_params.copy()
        else:
            self.defaults = defaults.copy()
        self.options = options.copy()

        self.bar_spec = self.defaults | self.options
        self.file = None
        debug = self.options.pop('debug', None) or DEBUG

    def __repr__(self) -> str:
        cls = type(self).__name__
        file = self.file
        bar_spec = self.bar_spec
        return f"<{cls} {f'{file=}, ' if file else ''}{bar_spec=}>"

    @classmethod
    def from_file(cls,
        file: os.PathLike = None,
        overrides: TemplateSpec = {},
        defaults: TemplateSpec = None,
    ) -> 'Template' | NoReturn:
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
        cfg = cls(options, defaults)
        cfg.file = absolute
        return cfg

    @staticmethod
    def read_file(file: os.PathLike) -> tuple[TemplateSpec, JSONText]:
        '''
        '''
        with open(file, 'r') as f:
            data = json.load(f)
            text = f.read()
        return data, text

    @staticmethod
    def write_file(
        file: os.PathLike,
        obj: TemplateSpec = {},
        defaults: BarParamSpec = None
    ) -> None:
        '''
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


def run(once: bool = False) -> None:
    '''Generate a bar from the default config file and run it in STDOUT.
    '''
    # bar = Bar.from_dict(Bar._default_params)
    cfg = Template.from_file(CONFIG_FILE)
    bar = Bar.from_dict(cfg.bar_spec)
    bar.run(once=once)

