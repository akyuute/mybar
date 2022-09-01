#!/usr/bin/python

#TODO: Command line args!
#TODO: Finish Mocp line!
#TODO: Implement killing threads!
#TODO: Implement align_to_seconds!
#TODO: Implement dynamic icons!


import asyncio
import inspect
import json
import os
import psutil
import signal
import sys
import threading
import time

from argparse import ArgumentParser
from string import Formatter
from typing import Iterable, Callable, IO

from field_funcs import *


CONFIG_FILE = 'bar.json' #'~/bar.json'
CSI = '\033['  # Unix terminal escape code (control sequence introducer)
CLEAR_LINE = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR = '?25l'
UNHIDE_CURSOR = '?25h'
COUNT = [0]


class InvalidOutputStream(AttributeError):
    '''Raised when an IO stream lacks write(), flush() and isatty() methods.'''
    pass
class IncompatibleParams(ValueError):
    '''Raised when a class is instantiated with one or more missing or
    incompatible parameters.'''
    pass
class BadFormatString(ValueError):
    '''Raised when a format string cannot be properly parsed or contains
    positional fields ('{}').'''
    pass
class InvalidField(TypeError):
    '''Raised when a field is either not an instance of Field or a string not
    found in the default fields collection.'''
    pass
class MissingBar(AttributeError):
    '''Raised when Field.run() is called before its instance is passed to the
    fields parameter in Bar().'''
    pass
class DefaultNotFound(NameError):
    '''Raised for references to an undefined default Field or function.'''
    pass
class UndefinedField(NameError):
    '''Raised if, when parsing a config file, a field name appears in the
    'field_order' item of the dict passed to Bar.from_dict() that is neither
    found in its 'field_definitions' parameter nor in Field._default_fields.'''
    pass


class Field:

    _default_fields = {
        'hostname': {
            'name': 'hostname',
            'func': get_hostname,
            'interval': 10
        },
        'uptime': {
            'name': 'uptime',
            'func': get_uptime,
            'kwargs': {'fmt': '%-jd:%-Hh:%-Mm'},
            'term_icon': 'Up '
        },
        'cpu_usage': {
            'name': 'cpu_usage',
            'func': get_cpu_usage,
            'interval': 2,
            'threaded': True
        },
        'cpu_temp': {
            'name': 'cpu_temp',
            'func': get_cpu_temp,
            'interval': 2,
            'threaded': True,
            'term_icon': 'CPU '
        },
        'mem_usage': {
            'name': 'mem_usage',
            'func': get_mem_usage,
            'interval': 2,
            'term_icon': 'Mem '
        },
        'disk_usage': {
            'name': 'disk_usage',
            'func': get_disk_usage,
            'interval': 4,
            'term_icon': '/:'
        },
        'battery': {
            'name': 'battery',
            'func': get_battery_info,
            'interval': 1,
            'overrides_refresh': True,
            'term_icon': 'Bat '
        },
        'net_stats': {
            'name': 'net_stats',
            'func': get_net_stats,
            'interval': 4
        },
        'datetime': {
            'name': 'datetime',
            'func': get_datetime,
            'interval': 0.125,
            'overrides_refresh': True
        }
    }

    def __init__(self,
        /,
        name: str = None,
        func: Callable = None,
        icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        # align_to_seconds: bool = False,
        overrides_refresh: bool = False,
        threaded: bool = False,
        constant_output: str = None,
        run_once: bool = False,
        bar=None,
        # args = [],
        args = None,
        # kwargs = {},
        kwargs = None,
        gui_icon: str = None,
        term_icon: str = None,
    ):

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
        self.args = args
        self.kwargs = kwargs

        if fmt is None:
            if all(s is None for s in (gui_icon, term_icon, icon)):
                raise IncompatibleParams(
                    "An icon is required when fmt is None.")
        self.fmt = fmt

        self._bar = bar

        self.term_icon = term_icon
        self.gui_icon = gui_icon
        self.icon = self.get_icon(icon)
        self.default_icon = icon

        if inspect.iscoroutinefunction(func) or threaded:
            self._callback = func
        else:
            # Wrap a synchronous function call
            self._callback = self._asyncify

        # self.align_to_seconds = align_to_seconds
        self._buffer = None
        self.constant_output = constant_output
        self.run_once = run_once
        self.interval = interval
        self.overrides_refresh = overrides_refresh

        self.threaded = threaded
        self._thread = None
        # self.is_running = False

    @classmethod
    def from_default(cls,
        name: str,
        params: dict = {},
        source: dict = None
    ):
        if source is None:
            source = cls._default_fields

        default: dict = source.get(name)
        if default is None:
            raise DefaultNotFound(
                f"{name!r} is not the name of a default Field.")

        spec = {**default}
        spec.update(params)
        return cls(**spec)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

    def get_icon(self, default=None):
        if self._bar is None:
            return default
        icon = self.term_icon if self._bar._stream.isatty() else self.gui_icon
        if icon is None:
            icon = default
        return icon

    async def _asyncify(self, *args, **kwargs):
        '''Wrap a synchronous function in a coroutine for simplicity.'''
        return self._func(*args, **kwargs)

    async def run(self):
        '''Asynchronously run a non-threaded field's callback
        and send updates to the bar.'''
        self._check_bar()
        bar = self._bar
        running = bar._can_run
        run_once = self.run_once
        override_queue = bar._override_queue
        overrides_refresh = self.overrides_refresh
        # self.is_running = True

        field_name = self.name
        field_buffers = bar._buffers
        interval = self.interval
        func = self._callback
        # args = self.args
        args = tuple() if self.args is None else self.args
        # kwargs = self.kwargs
        kwargs = {} if self.kwargs is None else self.kwargs

        icon = self.get_icon(self.default_icon)
        fmt = self.fmt
        use_format_str = (fmt is not None)
        last_val = None

        while running.is_set():
            res = await func(*args, **kwargs)

            if res == last_val:
                await asyncio.sleep(interval)
                continue

            last_val = res

            if use_format_str:
                contents = fmt.format(res, icon=icon)
            else:
                contents = icon + res
            field_buffers[field_name] = contents

            # Send new field contents to the bar's override queue and print a
            # new line between refresh cycles.
            if overrides_refresh:
                try:

                    # print(f"{(self._bar._loop == asyncio.get_running_loop()) = }")
                    override_queue.put_nowait(
                        (field_name, contents)
                    )
                except asyncio.QueueFull:
                    # Since the bar buffer was just updated, do nothing if the
                    # queue is full. The update may still show while the queue
                    # handles the current override.
                    # If not, the line will update at the next refresh cycle.
                    pass

                # Running the bar a second time raises RuntimeError saying its
                # event loop is closed.
                # It's not, so we ignore that.
                except RuntimeError:
                    # print("\nGot RuntimeError")
                    assert self._bar._loop.is_running()
                    pass

            if run_once:
                break

            await asyncio.sleep(interval)

    def run_threaded(self):
        '''Run a blocking function in a thread
        and send its updates to the bar.'''
        self._check_bar()
        bar = self._bar
        running = bar._can_run
        run_once = self.run_once
        override_queue = bar._override_queue
        overrides_refresh = self.overrides_refresh
        # self.is_running = True

        field_name = self.name
        field_buffers = bar._buffers
        interval = self.interval
        func = self._callback
        # args = self.args
        args = tuple() if self.args is None else self.args
        # kwargs = self.kwargs
        kwargs = {} if self.kwargs is None else self.kwargs

        icon = self.get_icon(self.default_icon)
        fmt = self.fmt
        use_format_str = (fmt is not None)
        last_val = None

        cooldown = bar._thread_cooldown

        # If the field's callback is asynchronous, run it in an event loop.
        is_async = inspect.iscoroutinefunction(func)
        loop = asyncio.new_event_loop()

        if self.run_once:
            if is_async:
                res = loop.run_until_complete(func(*args, **kwargs))
            else:
                res = func(*args, **kwargs)

            if use_format_str:
                contents = fmt.format(res, icon=icon)
            else:
                contents = icon + res
            field_buffers[field_name] = contents
            return

        count = 0
        while running.is_set():

            # Instead of blocking the entire interval, use a quicker cooldown.
            # A shorter cooldown means more chances to check if the bar stops.
            # A thread, then, cancels cooldown seconds after its function
            # returns rather than interval seconds.
            if count != int(interval / cooldown):
                count += 1
                time.sleep(cooldown)
                continue

            if is_async:
                res = loop.run_until_complete(func(*args, **kwargs))
            else:
                res = func(*args, **kwargs)

            if res == last_val:
                count = 0
                continue

            last_val = res

            if use_format_str:
                contents = fmt.format(res, icon=icon)
            else:
                contents = icon + res
            field_buffers[field_name] = contents

            # Send new field contents to the bar's override queue and print a
            # new line between refresh cycles.
            if overrides_refresh:
                try:
                    override_queue.put_nowait(
                        (field_name, contents)
                    )
                except asyncio.QueueFull:
                    # Since the bar buffer was just updated, do nothing if the
                    # queue is full. The update may still show while the queue
                    # handles the current override.
                    # If not, the line will update at the next refresh cycle.
                    pass

            if run_once:
                break

            count = 0

        loop.stop()
        loop.close()

    async def send_to_thread(self):
        '''Make and start a thread in which to run the field's callback.'''
        self._thread = threading.Thread(
            target=self.run_threaded,
            name=self.name
        )
        self._thread.start()

    def _check_bar(self):
        '''Raises MissingBar if self._bar is None.'''
        if self._bar is None:
            raise MissingBar("Fields cannot run until they belong to a Bar.")

class Bar:
    _field_funcs = {
        'hostname': get_hostname,
        'uptime': get_uptime,
        'cpu_usage': get_cpu_usage,
        'cpu_temp': get_cpu_temp,
        'mem_usage': get_mem_usage,
        'disk_usage': get_disk_usage,
        'battery': get_battery_info,
        'net_stats': get_net_stats,
        'datetime': get_datetime,
    }

    _default_fields = {
        'hostname': Field(name='hostname', func=get_hostname, interval=10),
        'uptime': Field(name='uptime', func=get_uptime, kwargs={'fmt': '%-dd:%-Hh:%-Mm'}),
        'cpu_usage': Field(name='cpu_usage', func=get_cpu_usage, interval=2, threaded=True),
        'cpu_temp': Field(name='cpu_temp', func=get_cpu_temp, interval=2, threaded=True),
        'mem_usage': Field(name='mem_usage', func=get_mem_usage, interval=2),
        'disk_usage': Field(name='disk_usage', func=get_disk_usage, interval=10),
        'battery': Field(name='battery', func=get_battery_info, interval=1, overrides_refresh=True),
        'net_stats': Field(name='net_stats', func=get_net_stats, interval=10),
        'datetime': Field(name='datetime', func=get_datetime, interval=1/8),
    }

    def __init__(self,
        *,
        fields: Iterable[Field|str] = None,
        fmt: str = None,
        separator: str = '|',
        refresh_rate: float = 1.0,
        stream: IO = sys.stdout,
        show_empty_fields: bool = False,
        gui_sep: str = None,
        term_sep: str = None,
        override_cooldown: float = 1/60,
        thread_cooldown: float = 1/8
    ):

        io_methods = ('write', 'flush', 'isatty')
        if not all(hasattr(stream, a) for a in io_methods):
            io_method_calls = [a + '()' for a in io_methods]
            raise InvalidOutputStream(
                f"Output stream {stream!r} needs "
                f"{join_options(io_method_calls, final=' and ')} methods.")
        self._stream = stream

        if fmt is None:
            if fields is None:
                raise ValueError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required.")

            if not hasattr(fields, '__iter__'):
                raise ValueError("The 'fields' argument must be iterable.")

            if all(s is None for s in (gui_sep, term_sep, separator)):
                raise IncompatibleParams(
                    "A separator is required when 'fmt' is None.")

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
            fields = list(field_order)

        self._field_order = field_order
        self.fmt = fmt
        self.term_sep = term_sep
        self.gui_sep = gui_sep
        self.separator = self.get_separator(separator)
        self.default_sep = separator

        self._fields = self.convert_fields(fields)
        self._buffers = {name: '' for name in self._fields}

        # Whether empty fields are shown with the rest when fmt is None:
        self.show_empty_fields = show_empty_fields

        # Set the bar's normal refresh rate, that is, how often it is printed:
        self.refresh_rate = refresh_rate

        # Make a queue to which fields with overrides_refresh send updates.
        # Process only one item per refresh cycle to reduce flickering in a GUI.
        self._override_queue = asyncio.Queue(maxsize=1)
        # How long should the queue checker sleep before processing a new item?
        # (A longer cooldown means less flickering):
        self._override_cooldown = override_cooldown

        # Staggering a thread's refresh interval across several sleeps with the
        # length of this cooldown prevents it from blocking for the entire
        # interval when the bar stops.
        # (A shorter cooldown means faster exits):
        self._thread_cooldown = thread_cooldown

        # Setting this Event cancels all fields:
        self._can_run = threading.Event()

        # The bar's async event loop:
        self._loop = asyncio.new_event_loop()

    @property
    def in_a_tty(self) -> bool:
        if self._stream is None:
            return False
        return self._stream.isatty()

    @property
    def clearline_char(self) -> str:
        if self._stream is None:
            return None
        clearline = CLEAR_LINE if self._stream.isatty() else ''
        return clearline

    @classmethod
    def from_dict(cls, dct: dict):
        data = dct.copy()
        field_dicts = data.pop('field_definitions', None)
        bar_params = data

        if (field_order := bar_params.pop('field_order', None)) is None:
            if (fmt := bar_params.get('fmt')) is None:
                raise IncompatibleParams(
                    "A bar format string 'fmt' is required when field order "
                    "list 'field_order' is undefined."
                )
            field_order = cls.parse_fmt(fmt)

        fields = []
        for name in field_order:
            params = field_dicts.get(name)

            if isinstance(params, dict):
                # Disallow 'name' param if coming from a dict anyway:
                #TODO: Consider raising an error!
                params.pop('name', None)

            if params is None:
                # The field is strictly default:
                field = Field.from_default(name)
                if field is None:
                    raise UndefinedField(f"\n"
                        f"In 'field_order':\n"
                        f"  Expected the name of a default field or a custom "
                        f"field defined in 'field_definitions', "
                        f"but got {name!r} instead."
                    )

            elif params.pop('custom', False):
                # The field is only defined in field_definitions and will not
                # be found as a default.
                field = Field(**params, name=name)

            else:
                # The field is a default overridden by the user:
                params = field_dicts.get(name)
                field = Field.from_default(name, params)
                if field is None:
                    raise DefaultNotFound(f"\n"
                        f"In 'field_definitions':\n"
                        f"  Expected a default Field name but got {name!r} "
                        f"instead.\n"
                        f"  Remember to specify `\"custom\": true` "
                        f"in custom field definitions."
                    )

            fields.append(field)

        bar = cls(fields=fields, **bar_params)
        return bar

    @staticmethod
    def parse_fmt(fmt: str) -> list[str]:
        '''Returns a list of field names that should act as a field order.'''
        try:
            field_names = [
                name
                for m in Formatter().parse(fmt)
                if (name := m[1]) is not None
            ]
        except ValueError:
            e = BadFormatString(f"Invalid bar format string: {fmt!r}")
            raise e from None
        if '' in field_names:
            raise BadFormatString(
                "The bar format string cannot contain "
                "positional fields ('{}').")
        return field_names

    def convert_fields(self, fields: Iterable[str] = None) -> dict[str, Field]:
        '''Converts strings in a list of fields to corresponding default Fields
        and returns a dict mapping field names to Fields.'''
        converted = {}
        for field in fields:
            if isinstance(field, str):
                converted[field] = Field.from_default(
                    name=field,
                    params={'bar': self}
                )
            elif isinstance(field, Field):
                field._bar = self
                converted[field.name] = field
            else:
                raise InvalidField(f"Invalid field: {field}")
        return converted

    def get_separator(self, default=None) -> str:
        if self._stream is None:
            return default
        sep = self.term_sep if self._stream.isatty() else self.gui_sep
        if sep is None:
            sep = default
        return sep

    def run(self, stream: IO = None):
        '''Run the bar. Block until an exception is raised and exit smoothly.'''
        if stream is not None:
            self._stream = stream

        # Allow the bar to run repeatedly in the same interpreter.
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        try:
            self._can_run.set()
            self._loop.run_until_complete(self._startup())

        except KeyboardInterrupt:
            pass

        finally:
            if self.in_a_tty:
                self._stream.write('\n')
                self._stream.write(CSI + UNHIDE_CURSOR)
            self._shutdown()

    async def _startup(self):
        '''Schedule field coroutines, threads and the line printer to be run
        in parallel.'''
        field_coros = []
        for field in self._fields.values():
            # Do not run fields which have a constant output;
            # only set the bar buffer.
            if field.constant_output is not None:
                if field.fmt is None:
                    self._buffers[field.name] = field.constant_output
                else:
                    self._buffers[field.name] = field.fmt.format(
                        field.constant_output
                    )
                continue

            if field.threaded:
                await field.send_to_thread()
            else:
                field_coros.append((field.run()))

        await asyncio.gather(
            *field_coros,
            self._check_queue(),
            self._continuous_line_printer(),
        )

    def _shutdown(self):
        '''Notify fields that the bar has stopped,
        close the event loop and join threads.'''
        self._can_run.clear()
        self._loop.stop()
        self._loop.close()
        for field in self._fields.values():
            if field.threaded and field._thread is not None:
##                # Optionally kill threads that are still blocking program exit.
##                if self._kill_threads:
##                    signal.pthread_kill(field._thread.ident, self._kill_signal)
##                    print(f"Sent {self._kill_signal} to {field.name}")
                field._thread.join()
                # print(f"{field._thread.name}: {field._thread.is_alive() = }")

    async def _continuous_line_printer(self, end: str = '\r'):
        '''The bar's primary line-printing mechanism.
        Fields are responsible for sending updates to the bar's buffers.
        This only writes using the current buffer contents.'''
        use_format_str = (self.fmt is not None)
        stream = self._stream
        sep = self.separator
        clearline = self.clearline_char
        show_empty_fields = self.show_empty_fields

        if self.in_a_tty:
            stream.write(CSI + HIDE_CURSOR)
            beginning = clearline + end
        else:
            beginning = clearline

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        stream.flush()
        start_time = time.monotonic_ns()
        while self._can_run.is_set():

            if use_format_str:
                line = self.fmt.format_map(self._buffers)
            else:
                line = sep.join(
                    buf
                    for field in self._field_order
                        if (buf := self._buffers[field])
                        or show_empty_fields
                )

            stream.write(beginning + line + end)
            stream.flush()

            # Syncing the refresh rate to the system clock prevents drifting.
            await asyncio.sleep(
                self.refresh_rate - (
                    (time.monotonic_ns() - start_time)
                    % self.refresh_rate
                )
            )

    async def _check_queue(self, end: str = '\r'):
        '''Prints a line when fields with overrides_refresh send new data.'''
        use_format_str = (self.fmt is not None)
        stream = self._stream
        sep = self.separator
        clearline = self.clearline_char
        show_empty_fields = self.show_empty_fields
        cooldown = self._override_cooldown

        if self.in_a_tty:
            beginning = clearline + end
        else:
            beginning = clearline

        try:
            while self._can_run.is_set():

                field, contents = await self._override_queue.get()

                if use_format_str:
                    line = self.fmt.format_map(self._buffers)
                else:
                    line = sep.join(
                        buf
                        for field in self._field_order
                            if (buf := self._buffers[field])
                            or show_empty_fields
                    )

                stream.write(beginning + line + end)
                stream.flush()
                await asyncio.sleep(cooldown)

        # SIGINT raises RuntimeError saying the event loop is closed.
        # It's not, so we ignore that.
        except RuntimeError:
            assert self._loop.is_running()
            pass

class Config:
    def __init__(self, file: os.PathLike = None): # Path = None):
        # Get the config file name if passed as a command line argument
        cli_parser = ArgumentParser()
        cli_parser.add_argument('--config', nargs='+')
        config_file = cli_parser.parse_args(sys.argv[1:]).config or file

        if config_file is None:
            config_file = os.path.expanduser(CONFIG_FILE)

        self.file = config_file
        with open(self.file, 'r') as f:
            self.data = json.load(f)
            self.raw = f.read()

    def write_file(self, file: os.PathLike = None, obj: dict = None):
        if file is None:
            file = self.file
        if obj is None:
            obj = self.data
        # return json.dumps(obj)
        with open(file, 'w') as f:
            json.dump(obj, f)


    def get_bar(self) -> Bar:
        return Bar.from_dict(self.data)

    def _make_error_message(self,
        label: str,
        blame = None,
        expected: str = None,
        file: str = None,
        line: int = None,
        indent: str = "  ",
        details: Iterable[str] = None
    ) -> str:
        level = 0

        message = []
        if file is not None:
            message.append(f"In config file {file!r}")
            if line is not None:
                message[-1] += f" (line {line})"
            message[-1] += ":"
            level += 1

        elif line is not None:
            message.append(f"(line {line}):")
            level += 1

        message.append(f"{indent * level}While parsing {label}:")
        level += 1

        if details is not None:
            message.append('\n'.join((indent * level + det) for det in details))
            # message.append(
                # ('\n' + indent * level).join(details)
            # )

        if blame is not None:
            if expected is not None:
                message.append(
                    f"{indent * level}Expected {expected}, "
                    f"but got {blame} instead."
                )
            else:
                message.append(f"{indent * level}{blame}")

        err = ('\n').join(message)
        return err

def main():
    fhostname = Field(name='hostname', func=get_hostname, run_once=True, term_icon='')
    fuptime = Field(name='uptime', func=get_uptime, kwargs={'fmt': '%-jd:%-Hh:%-Mm'}, term_icon='Up:')
    fcpupct = Field(name='cpu_usage', func=get_cpu_usage, interval=2, threaded=True, term_icon='CPU:')
    fcputemp = Field(name='cpu_temp', func=get_cpu_temp, interval=2, threaded=True, term_icon='')
    fmem = Field(name='mem_usage', func=get_mem_usage, interval=2, term_icon='Mem:')
    fdisk = Field(name='disk_usage', func=get_disk_usage, interval=2, term_icon='/:')
    # fbatt = Field(name='battery', func=get_battery_info, interval=1, overrides_refresh=False, term_icon='Bat:')
    fbatt = Field(name='battery', func=get_battery_info, interval=1, overrides_refresh=True, term_icon='Bat:')
    fnet = Field(name='net_stats', func=get_net_stats, interval=4, term_icon='', threaded=False)

    fdate = Field(
        # fmt="<3 {icon}{}!",
        name='datetime',
        func=get_datetime,
        interval=1/8,
        overrides_refresh=True,
        term_icon='?'
    )
    fcounter = Field(name='counter', func=counter, interval=1, args=[COUNT])

    # fdate = Field(name='datetime', func=get_datetime, interval=1, overrides_refresh=False, term_icon='')
    # fdate = Field(func=get_datetime, interval=0.5, overrides_refresh=False)
    # fdate = Field(func=get_datetime, interval=1)
    # mocpline = Field(name='mocpline', func=

    global fields
    fields = (
        Field(
            name='isatty',
            func=(lambda args=None, kwargs=None: str(bar.in_a_tty)),
            # icon='TTY:',
            run_once=True,
            # interval=0
        ),
        # fhostname,
        # fuptime,
        fcpupct,
        fcputemp,
        fmem,
        fdisk,
        fbatt,
        fnet,
        fdate,
        # fcounter,
    )

    # fields = [Field(name='test', interval=1/16, func=(lambda args=None, kwargs=None: "WORKING"), overrides_refresh=True, term_icon='&', gui_icon='@')]

    global bar
    # bar = Bar(fields=fields)
    # bar = Bar(fields=fields, sep=None, fmt=None)  # Raise IncompatibleParams

    # fmt = "Up{uptime} | CPU: {cpu_usage}, {cpu_temp}|Disk: {disk_usage} Date:{datetime}..."
    # fmt = None
    # bar = Bar(fmt=fmt)

    # bar = Bar(fields=[fdate, fcount, fhostname, fuptime])

    # bar = Bar(fields='uptime cpu_usage cpu_temp mem_usage datetime datetime datetime disk_usage battery net_stats datetime'.split())


    global CFG
    # CFG = Config('bar.conf')
    CFG = Config('bar.json')
    bar = CFG.get_bar()

##    if not os.path.exists(file):
##        CFG.write_config(file, defaults)

    bar.run()


if __name__ == '__main__':
    main()

