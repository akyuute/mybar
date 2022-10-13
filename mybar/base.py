#TODO: Re-DRY everything (inline later if appropriate)
#TODO: collections.defaultdict, dict.fromkeys!
#TODO: Command line args!
#TODO: Finish Mocp line!
#TODO: Implement dynamic icons!


__all__ = (
    'Field',
    'Bar',
    'Config',
    'run',
)


import asyncio
import inspect
import json
import logging
import os
import sys
import threading
import time
from argparse import ArgumentParser
from string import Formatter

from mybar import field_funcs
from mybar import setups
from mybar.errors import *

from mybar.utils import (
    join_options,
    str_to_bool,
    scrub_comments,
    make_error_message
)


# Typing
from collections.abc import Callable, Iterable, Sequence
from typing import IO

BarSpec = dict
FieldSpec = dict

PTY_Icon = str
TTY_Icon = str
PTY_Separator = str
TTY_Separator = str

ConsoleControlCode = str
FormatStr = str

Args = list
Kwargs = dict


CONFIG_FILE = '~/.mybar.json'

# Unix terminal escape code (control sequence introducer):
CSI: ConsoleControlCode = '\033['
CLEAR_LINE: ConsoleControlCode = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR: ConsoleControlCode = '?25l'
UNHIDE_CURSOR: ConsoleControlCode = '?25h'

DEBUG = False
logging.basicConfig(
    level='DEBUG',
    filename=os.path.expanduser('~/.mybar.log'),
    filemode='w',
    datefmt='%Y-%m-%d_%H:%M:%S.%f',
    format='[{asctime}] ({levelname}:{name}) {message}',
    style='{',
)
logger = logging.getLogger(__name__)


class Field:

    _default_fields = {

        'hostname': {
            'name': 'hostname',
            'func': field_funcs.get_hostname,
            'interval': 10
        },

        'uptime': {
            'name': 'uptime',
            'func': field_funcs.get_uptime,
            'setup': setups.setup_uptime,
            'kwargs': {
                'fmt': '{days}d:{hours}h:{mins}m',
                'sep': ':'
            },
            'align_to_seconds': True,
            'icon': 'Up ',
        },

        'cpu_usage': {
            'name': 'cpu_usage',
            'func': field_funcs.get_cpu_usage,
            'interval': 2,
            'threaded': True,
            'icon': 'CPU ',
        },

        'cpu_temp': {
            'name': 'cpu_temp',
            'func': field_funcs.get_cpu_temp,
            'interval': 2,
            'threaded': True
        },

        'mem_usage': {
            'name': 'mem_usage',
            'func': field_funcs.get_mem_usage,
            'interval': 2,
            'icon': 'Mem '
        },

        'disk_usage': {
            'name': 'disk_usage',
            'func': field_funcs.get_disk_usage,
            'interval': 4,
            'icon': '/:'
        },

        'battery': {
            'name': 'battery',
            'func': field_funcs.get_battery_info,
            'icon': 'Bat '
        },

        'net_stats': {
            'name': 'net_stats',
            'func': field_funcs.get_net_stats,
            'interval': 4
        },

        'datetime': {
            'name': 'datetime',
            # 'func': field_funcs.precision_datetime,
            'func': field_funcs.get_datetime,
            'kwargs': {
                'fmt': "%Y-%m-%d %H:%M:%S"
            },
            'align_to_seconds': True
        },

    }

    def __init__(self,
        *,
        name: str = None,
        func: Callable[..., str] = None,
        icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        align_to_seconds: bool = False,
        overrides_refresh: bool = False,
        threaded: bool = False,
        # wrap: bool = True,
        constant_output: str = None,
        run_once: bool = False,
        bar: 'Bar' = None,
        args = None,
        kwargs = None,
        setup: Callable[..., Kwargs] = None,

        # Set this to use different icons for different output streams:
        icons: Sequence[PTY_Icon, TTY_Icon] = None,
    ) -> None:

        if constant_output is None:
            #NOTE: This will change when dynamic icons and fmts are implemented.
            if func is None:
                raise IncompatibleParams(
                    f"Either a function that returns a string or "
                    f"a constant output string is required."
                )
            if not callable(func):
                raise TypeError(
                    f"Type of 'func' must be callable, not {type(func)}")

        if name is None and callable(func):
            name = func.__name__
        self.name = name

        self._func = func
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

        if setup is not None and not callable(setup):
            raise TypeError(
                f"Parameter 'setup' must be a callable, not {type(setup)}")
        self._setupfunc = setup
        # self._setupvars = {}

        if icons is None:
            icons = (icon, icon)
        self._icons = icons

        if fmt is None and icons is None:
            raise IncompatibleParams("An icon is required when fmt is None.")
        self.fmt = fmt

        self._bar = bar

        if inspect.iscoroutinefunction(func) or threaded:
            self._callback = func
        else:
            # Wrap a synchronous function call:
            self._callback = self._asyncify

        self.align_to_seconds = align_to_seconds
        self._buffer = None
        self.constant_output = constant_output
        self.interval = interval
        self.overrides_refresh = overrides_refresh
        self.run_once = run_once

        self.threaded = threaded
        self._thread = None
        # self.is_running = False

    def __repr__(self) -> str:
        cls = type(self).__name__
        name = self.name
        # attrs = join_options(...)
        return f"{cls}({name=})"

    @classmethod
    def from_default(cls,
        name: str,
        params: FieldSpec = {},
        source: dict = None
    ):
        '''Used to create default Fields with custom parameters.'''
        if source is None:
            source = cls._default_fields

        default: dict = source.get(name)
        if default is None:
            return default
            raise DefaultFieldNotFound(
                f"{name!r} is not the name of a default Field.")

        spec = {**default}
        spec.update(params)
        return cls(**spec)

    @property
    def icon(self) -> str:
        '''The field icon as determined by the output stream of its bar.
        It defaults to the TTY icon (self._icons[1]) if no bar is set.
        '''
        if self._bar is None:
            return self._icons[1]  # Default to using the terminal icon.
        return self._icons[self._bar._stream.isatty()]

    async def _asyncify(self, *args, **kwargs) -> str:
        '''Wrap a synchronous function in a coroutine for simplicity.'''
        return self._func(*args, **kwargs)

    @staticmethod
    def _format_contents(text: str, icon: str, fmt: FormatStr) -> str:
        '''A helper function that formats field contents.'''
        if fmt is None:
            return icon + text
        else:
            return fmt.format(text, icon=icon)

    async def run(self, once: bool = False) -> None:
        '''Asynchronously run a non-threaded field's callback
        and send updates to the bar.'''
        self._check_bar()

        # Defining local variables imperceptibly improves performance.
        # (Fewer LOAD_ATTRs, more LOAD_FASTs.)
        bar = self._bar
        running = bar._can_run.is_set
        override_queue = bar._override_queue
        overrides_refresh = self.overrides_refresh
        # self.is_running = True

        field_name = self.name
        field_buffers = bar._buffers
        interval = self.interval
        clock = time.monotonic

        func = self._callback
        args = self.args
        kwargs = self.kwargs

        icon = self.icon
        fmt = self.fmt
        using_format_str = (fmt is not None)
        last_val = None

        # Use self.setup() to gather any static variables which need to
        # be evaluated at runtime and passed to self._func.
        if self._setupfunc is not None:
            try:
                kwargs.update(
                    await self.setup(args, kwargs)
                )

            except FailedSetup as e:
                backup = e.args[0]
                contents = self._format_contents(backup, icon, fmt)
                field_buffers[field_name] = self.constant_output = contents
                return

        # Run at least once at the start to ensure the bar is not empty:
        result = await func(*args, **kwargs)
        last_val = result
        contents = self._format_contents(result, icon, fmt)
        field_buffers[field_name] = contents

        if self.run_once or once:
            return

        if self.align_to_seconds:
            # Sleep until the beginning of the next second.
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))
        start_time = clock()

        # The main loop:
        while running():
            result = await func(*args, **kwargs)
            # Latency from nonzero execution times causes drift, where
            # sleeps become out-of-sync and the bar skips field updates.
            # This is especially noticeable in fields that update
            # routinely, such as the time.

            # To negate drift, when sleeping until the beginning of the
            # next cycle, we must also compensate for latency.
            await asyncio.sleep(
                # Time until next refresh:
                interval - (
                    # Get the current latency, which can vary:
                    clock() % interval
                    # (clock() - start_time) % interval  # Preserve offset
                )
            )

            # if DEBUG:
                # logger.debug(f"{field_name}: New refresh cycle at {clock() - start_time}")

            if result == last_val:
                continue
            last_val = result

            if using_format_str:
                contents = fmt.format(result, icon=icon)
            else:
                contents = icon + result
            field_buffers[field_name] = contents

            # Send new field contents to the bar's override queue and print a
            # new line between refresh cycles.
            if overrides_refresh:
                try:
                    override_queue.put_nowait((field_name, contents))

                except asyncio.QueueFull:
                    # Since the bar buffer was just updated, do nothing if the
                    # queue is full. The update may still show while the queue
                    # handles the current override.
                    # If not, the line will update at the next refresh cycle.
                    pass

    def run_threaded(self, once: bool = False) -> None:
        '''Run a blocking function in a thread
        and send its updates to the bar.'''
        self._check_bar()

        # Defining local variables imperceptibly improves performance.
        # (Fewer LOAD_ATTRs, more LOAD_FASTs.)
        bar = self._bar
        running = bar._can_run.is_set
        override_queue = bar._override_queue
        overrides_refresh = self.overrides_refresh
        # self.is_running = True

        field_name = self.name
        field_buffers = bar._buffers
        interval = self.interval
        clock = time.monotonic
        step = bar._thread_cooldown

        func = self._callback
        args = self.args
        kwargs = self.kwargs

        icon = self.icon
        fmt = self.fmt
        using_format_str = (fmt is not None)
        last_val = None

        # If the field's callback is asynchronous, run it in an event loop.
        is_async = inspect.iscoroutinefunction(func)
        loop = asyncio.new_event_loop()

        # Use self.setup() to gather static variables which need to
        # be evaluated at runtime and passed to self._func.
        if self._setupfunc is not None:
            try:
                kwargs.update(
                    loop.run_until_complete(
                        self.setup(args, kwargs)
                    )
                )

            except FailedSetup as e:
                backup = e.args[0]
                contents = self._format_contents(backup, icon, fmt)
                field_buffers[field_name] = self.constant_output = contents
                return

        # Run at least once at the start to ensure the bar is not empty:
        if is_async:
            result = loop.run_until_complete(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        last_val = result
        contents = self._format_contents(result, icon, fmt)
        field_buffers[field_name] = contents

        if self.run_once or once:
            return

        count = 0
        needed = round(interval / step)
        # if DEBUG:
            # logger.debug(f"{field_name}: {needed = }")

        if self.align_to_seconds:
            # Sleep until the beginning of the next second.
            clock = time.time
            time.sleep(1 - (clock() % 1))
        start_time = clock()

        # The main loop:
        while running():
            # Rather than block for the whole interval,
            # use tiny steps to check if the bar is still running.
            # A shorter step means more chances to check if the bar stops.
            # Thus, threads usually cancel `step` seconds after `func`
            # returns when the bar stops rather than after `interval` seconds.

            # if DEBUG:
                # logger.debug(f"{field_name}: {count = }")
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
                        # (clock() - start_time) % step  # Preserve offset
                    )
                )
                continue

            # if DEBUG:
                # logger.debug(f"{field_name}: New refresh cycle at {clock() - start_time}")

            count = 0

            if is_async:
                result = loop.run_until_complete(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)

            if result == last_val:
                continue
            last_val = result

            if using_format_str:
                contents = fmt.format(result, icon=icon)
            else:
                contents = icon + result
            field_buffers[field_name] = contents

            # Send new field contents to the bar's override queue and print a
            # new line between refresh cycles.
            if overrides_refresh:
                # if DEBUG:
                    # logger.debug(f"{field_name}: Sending update @ {clock() - start_time}")
                try:
                    override_queue._loop.call_soon_threadsafe(
                        override_queue.put_nowait, (field_name, contents)
                    )

                except asyncio.QueueFull:
                    # if DEBUG:
                        # logger.debug(f"{field_name}: Failed update: QueueFull")
                    # Since the bar buffer was just updated, do nothing if the
                    # queue is full. The update may still show while the queue
                    # handles the current override.
                    # If not, the line will update at the next refresh cycle.
                    pass

        loop.stop()
        loop.close()

    def _check_bar(self) -> None:
        '''Raises MissingBar if self._bar is None.'''
        if self._bar is None:
            raise MissingBar("Fields cannot run until they belong to a Bar.")

    async def setup(self, args: Args, kwargs: Kwargs) -> dict:
        '''Initialize static variables used by self._func which would
        otherwise be evaluated at each iteration.'''
        if inspect.iscoroutinefunction(self._setupfunc):
            return await self._setupfunc(*args, **kwargs)
        return self._setupfunc(*args, **kwargs)

    async def send_to_thread(self, run_once: bool = True) -> None:
        '''Make and start a thread in which to run the field's callback.'''
        self._thread = threading.Thread(
            target=self.run_threaded,
            name=self.name,
            args=(run_once,)
        )
        self._thread.start()


class Bar:

    _default_params = {
        'separator': '|',
        'refresh_rate': 1.0,
    }

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
            raise InvalidOutputStream(
                f"Output stream {stream!r} needs "
                f"{join_options(io_method_calls, final_sep=' and ')} methods.")
        self._stream = stream

        # Ensure required parameters are defined if no fmt is given:
        if fmt is None:
            if fields is None:
                raise ValueError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required.")

            if not hasattr(fields, '__iter__'):
                raise ValueError("The 'fields' argument must be iterable.")

            if separator is None and separators is None:
                raise IncompatibleParams(
                    "A separator is required when 'fmt' is None.")

            # Gather a list of field names:
            field_order = [
                getattr(f, 'name') if isinstance(f, Field)
                else f
                for f in fields
            ]

        elif not isinstance(fmt, str):
            raise TypeError(
                f"Format string 'fmt' must be a string, not {type(fmt)}")

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

        # Calling run() sets this Event. Unsetting it stops all fields:
        self._can_run = threading.Event()

        # The bar's async event loop:
        self._loop = asyncio.new_event_loop()

    def __repr__(self) -> str:
        names = self._field_order
        fields = join_options(names, final_sep='', quote=True, limit=3)
        cls = type(self).__name__
        return f"{cls}(fields=[{fields}])"

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
    def clearline_char(self) -> str:
        '''A special character printed to TTY streams between refreshes.
        Its purpose is to clear characters left behind by longer lines.
        '''
        if self._stream is None:
            return None
        clearline = CLEAR_LINE if self._stream.isatty() else ''
        return clearline

    @property
    def separator(self) -> str:
        '''The field separator as determined by the output stream.
        It defaults to the TTY sep (self._separators[1]) if no stream is set.
        '''
        # return self._separator
        if self._stream is None:
            # Default to using the terminal separator:
            return self._separators[1]
        return self._separators[self._stream.isatty()]

    @classmethod
    def from_dict(cls, dct: dict):
        '''Accept a mapping of Bar parameters.'''
        #NOTE: This can raise AttributeError if it encounters a nested list!
        data = scrub_comments(dct, '//')
        field_defs = data.pop('field_definitions', None)
        bar_params = data
        field_order = bar_params.pop('field_order', None)

        if (fmt := bar_params.get('fmt')) is None:
            if field_order is None:
                raise IncompatibleParams(
                    "A bar format string 'fmt' is required when field "
                    "order list 'field_order' is undefined."
                )
        else:
            field_order = cls.parse_fmt(fmt)

        fields = []
        for name in field_order:
            params = field_defs.get(name)
            match params:

                case None:
                    # The field is strictly default:
                    field = Field.from_default(name)
                    if field is None:
                        exc = make_error_message(
                            UndefinedField,
                            whilst="parsing 'field_order'",
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
                    custom_name = params.pop('name', name)
                    del params['custom']
                    field = Field(**params, name=custom_name)

                case {}:
                    # The field is a default overridden by the user.
                    field = Field.from_default(name, params)
                    if field is None:

                        exc = make_error_message(
                            DefaultFieldNotFound,
                            whilst="parsing 'field_definitions'",
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
                        InvalidFieldSpec,
                        whilst="parsing 'field_definitions'",
                        details=(
                            f"Invalid Field specification: {params!r}",
                        ),
                        indent_level=1
                    )
                    raise exc

            fields.append(field)

        return cls(fields=fields, **bar_params)

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
            raise BrokenFormatString(err) from None
        if '' in field_names:
            raise BrokenFormatString(
                "The bar format string cannot contain "
                "positional fields ('{}').")
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
                    raise InvalidField(f"Invalid field: {field}")
        return converted

    def run(self, stream: IO = None, once: bool = None) -> None:
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
        self._active_queue_get_coro = None

        overriding = False
        gathered = []

        for field in self._fields.values():
            if field.overrides_refresh:
                overriding = True

            # Do not run fields which have a constant output;
            # only set their bar buffer.
            if field.constant_output is not None:
                self._buffers[field.name] = field._format_contents(
                    field.constant_output,
                    field.icon,
                    field.fmt
                )

            elif field.threaded:
                await field.send_to_thread(run_once)

            else:
                gathered.append(field.run(run_once))

        if run_once:
            await asyncio.gather(*gathered)
            self._print_one_line(self._make_one_line())

        else:
            if overriding:
                gathered.append(self._handle_overrides())
            await asyncio.gather(
                *gathered,
                self._continuous_line_printer()
            )

    def _make_one_line(self) -> str:
        '''Make a line using the bar's field buffers.
        This method is not meant to be called from within a loop.
        '''
        if self.fmt is not None:
            line = self.fmt.format_map(buffers)
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

    def _shutdown(self) -> None:
        '''Notify fields that the bar has stopped,
        close the event loop and join threads.'''
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
        stream = self._stream
        clearline = self.clearline_char
        using_format_str = (self.fmt is not None)
        fmt = self.fmt
        sep = self.separator
        buffers = self._buffers
        field_order = self._field_order
        join_empty_fields = self.join_empty_fields
        running = self._can_run.is_set
        refresh = self.refresh_rate
        clock = time.monotonic

        if self.in_a_tty:
            beginning = clearline + end
            stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = clearline

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        stream.flush()

        # Print something right away just so that the bar is not empty:
        if using_format_str:
            line = fmt.format_map(buffers)
        else:
            line = sep.join(
                buf
                for field in field_order
                    if (buf := buffers[field])
                    or join_empty_fields
            )
        stream.write(beginning + line + end)
        stream.flush()

        if self.align_to_seconds:
            # Begin every refresh at the start of a clock second:
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))

        start_time = clock()
        while running():
            if using_format_str:
                line = fmt.format_map(buffers)
            else:
                line = sep.join(
                    buf
                    for field in field_order
                        if (buf := buffers[field])
                        or join_empty_fields
                )

            stream.write(beginning + line + end)
            stream.flush()

            # Sleep only until the next possible refresh to keep the
            # refresh cycle length consistent and prevent drifting.
            await asyncio.sleep(
                # Time until next refresh:
                refresh - (
                    # Get the current latency, which can vary:
                    # clock() % refresh
                    (clock() - start_time) % refresh  # Preserve offset
                )
            )

    async def _handle_overrides(self, end: str = '\r') -> None:
        '''Prints a line when fields with overrides_refresh send new data.'''
        # Again, local variables may save time:
        stream = self._stream
        clearline = self.clearline_char
        using_format_str = (self.fmt is not None)
        fmt = self.fmt
        sep = self.separator
        buffers = self._buffers
        field_order = self._field_order
        join_empty_fields = self.join_empty_fields
        running = self._can_run.is_set
        refresh = self.refresh_rate
        override_queue = self._override_queue
        cooldown = self._override_cooldown

        if self.in_a_tty:
            beginning = clearline + end
        else:
            beginning = clearline

        start_time = time.time()
        while running():

            try:
                # Wait until a field with overrides_refresh sends new
                # data to be printed:
                field, contents = await override_queue.get()
                # if DEBUG:
                    # logger.debug(f"handler: {field} {time.time() - start_time}")

            except RuntimeError as exc:
                # asyncio raises RuntimeError if the event loop closes
                # while queue.get() is waiting for a value.
                # if DEBUG:
                    # logger.debug(exc.args[0])
                return

            if using_format_str:
                line = self.fmt.format_map(buffers)
            else:
                line = sep.join(
                    buf
                    for field in field_order
                        if (buf := buffers[field])
                        or join_empty_fields
                )

            stream.write(beginning + line + end)
            stream.flush()
            # if DEBUG:
                # logger.debug(f"handler: sleeping for {cooldown}")
            await asyncio.sleep(cooldown)

class Config:
    def __init__(self, file: os.PathLike = None) -> None:
        # Get the config file name if passed as a command line argument
        cli_parser = ArgumentParser()
        cli_parser.add_argument('--config')
        config_file = (
            cli_parser.parse_args(sys.argv[1:]).config
            or file
            or CONFIG_FILE
        )

        absolute = os.path.expanduser(config_file)
        if not os.path.exists(absolute):
            self.write_file(absolute)
        self.file = absolute

        self.data, self.text = self.read_file(absolute)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        file = self.file
        return f"{cls}({file=})"

    def make_bar(self) -> Bar:
        return Bar.from_dict(self.data)

    def read_file(self, file: os.PathLike = None) -> tuple[BarSpec, str]:
        if file is None:
            file = self.file
        with open(self.file, 'r') as f:
            data = json.load(f)
            text = f.read()
        return data, text

    def write_file(self,
        file: os.PathLike = None,
        obj: BarSpec = None
    ) -> None:
        if file is None:
            file = self.file

        # return json.dumps(obj)
        obj = Bar._default_params.copy() if obj is None else obj

        dft_bar = obj
        dft_fields = Field._default_fields.copy()

        for name, field in dft_fields.items():
            new = dft_fields[name] = field.copy()
            for param in ('name', 'func', 'setup'):
                try:
                    del new[param]
                except KeyError:
                    pass

        dft_bar['field_definitions'] = dft_fields
        dft_bar['field_order'] = Bar._default_field_order
        self.defaults = dft_bar

        # return self.defaults
        with open(os.path.expanduser(file), 'w') as f:
            json.dump(self.defaults, f, indent=4, ) #separators=(',\n', ': '))


def run() -> None:
    '''Generate a bar from the default config file and run it in STDOUT.
    '''

    CFG = Config(CONFIG_FILE)
##    if not os.path.exists(CONFIG_FILE):
##        CFG.write_file()

    bar = CFG.make_bar()
##    START = time.time()
##    refresh_test = Field(bar=bar, name='refresh_test', func=(lambda: str((time.time() - START)%1)), interval=0.1, threaded=True)
##    bar._fields['refresh_test'] = foo
##    bar._field_order.append('refresh_test')

    bar.run()

