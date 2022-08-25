#!/usr/bin/python

#TODO: Config files!
#TODO: Command line args!
#TODO: Finish Mocp line!
#TODO: Implement killing threads!
#TODO: Implement align_to_seconds!
#TODO: Implement dynamic icons!


import os
import sys
import time
import psutil
import signal
import asyncio
import inspect
import threading
import strictyaml
from string import Formatter
from argparse import ArgumentParser
from typing import Iterable, Callable, IO

from field_funcs import *


CSI = '\033['  # Unix terminal escape code (control sequence introducer)
CLEAR_LINE = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR = '?25l'
UNHIDE_CURSOR = '?25h'
COUNT = [0]


def join_options(
    it: Iterable[str],
    /,
    sep: str = ', ',
    final: str = ' or ',
    oxford: bool = False
):
    if not hasattr(it, '__iter__'):
        raise TypeError(f"Can only join an iterable, not {type(it)}.")
    return sep.join(list(it)[:-1]) + ('', ',')[oxford] + final + it[-1]

def str_to_bool(value: str, /):
    '''Returns `True` or `False` bools for truthy or falsy strings.'''
    truthy = "true t yes y on 1".split()
    falsy = "false f no n off 0".split()
    pattern = value.lower()
    if pattern not in truthy + falsy:
        raise ValueError(f"Invalid argument: {value!r}")
    return (pattern in truthy or not pattern in falsy)

def add_if_exists(dct: dict, key, val):
    if not isinstance(dct, dict):
        raise ValueError("dct must be a mutable mapping such as a dict.")
    if val is not None:
        dct[key] = val

class InvalidOutputStream(Exception):
    '''Raised when an IO stream lacks write(), flush() and isatty() methods.'''
    pass
class UndefinedSeparator(Exception):
    '''Raised when a Bar lacks a separator when its fmt is None.'''
    pass
class UndefinedIcon(Exception):
    '''Raised when a Field lacks an icon when its fmt is None.'''
    pass
class BadFormatString(Exception):
    '''Raised when a format string cannot be properly parsed or contains
    positional fields ('{}').'''
    pass
class InvalidField(Exception):
    '''Raised when a field is either not an instance of Field or a string not
    found in the default fields collection.'''
    pass
class MissingBar(Exception):
    '''Raised when Field.run() is called before its instance is passed to the
    fields parameter in Bar().'''
    pass
class DefaultNotFound(Exception):
    '''Raised for references to an undefined default Field or function.'''
    pass

class ConfigInvalid(Exception):
    '''Raised when a config file contains missing or malformed data.'''
    pass
class ConfigInvalidField(Exception):
    '''Raised when a field definition block of a config file is malformed.'''
    pass
class ConfigCannotConvert(Exception):
    '''Raised when a config value cannot be converted to the necessary type.'''
    pass


class Field:
    _default_fields = {
        'hostname': dict( #Field(
            name='hostname',
            func=get_hostname,
            interval=10
        # },
        ),
        'uptime': dict( #Field(
            name='uptime',
            func=get_uptime,
            kwargs={'fmt': '%-dd:%-Hh:%-Mm'}
        # },
        ),
        'cpu_usage': dict( #Field(
            name='cpu_usage',
            func=get_cpu_usage,
            interval=2,
            threaded=True
        # },
        ),
        'cpu_temp': dict( #Field(
            name='cpu_temp',
            func=get_cpu_temp,
            interval=2,
            threaded=True
        # },
        ),
        'mem_usage': dict( #Field(
            name='mem_usage',
            func=get_mem_usage,
            interval=2
        # },
        ),
        'disk_usage': dict( #Field(
            name='disk_usage',
            func=get_disk_usage,
            interval=10
        # },
        ),
        'battery': dict( #Field(
            name='battery',
            func=get_battery_info,
            interval=1,
            override_refresh_rate=True
        # },
        ),
        'net_stats': dict( #Field(
            name='net_stats',
            func=get_net_stats,
            interval=10
        # },
        ),
        'datetime': dict( #Field(
            name='datetime',
            func=get_datetime,
            interval=1/8,
            override_refresh_rate=True
        # },
        ),
    }

    def __init__(self,
        /,
        name: str = None,
        func: Callable = None,
        icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        align_to_seconds=False,
        override_refresh_rate=False,
        threaded=False,
        constant_output: str = None,
        run_once: bool = False,
        bar=None,
        args = [],
        kwargs = {},
        gui_icon: str = None,
        term_icon: str = None,
    ):
        if func is None and constant_output is None:
            raise ValueError(
                f"Either a function that returns a string or "
                f"a constant output string is required."
            )

        if not callable(func):
            raise TypeError(
                f"Type of 'func' must be callable, "
                f"not {type(self._callback)}"
            )
        self._func = func
        self.args = args
        self.kwargs = kwargs

        if fmt is None:
            if all(s is None for s in (gui_icon, term_icon, icon)):
                raise UndefinedIcon("An icon is required when fmt is None.")
        self.fmt = fmt

        self.bar = bar

        self.term_icon = term_icon
        self.gui_icon = gui_icon
        self.icon = self.get_icon(icon)
        self.default_icon = icon

        if inspect.iscoroutinefunction(self._func) or threaded:
            self._callback = self._func
        else:
            # Wrap a synchronous function call
            self._callback = self._asyncify

        if name is None:
            name = self._func.__name__
        self.name = name

        self.align_to_seconds = align_to_seconds
        self.buffer = None
        self.constant_output = constant_output
        self.run_once = run_once
        self.interval = interval
        self.overrides_refresh = override_refresh_rate

        self.is_threaded = threaded
        self.thread = None
        # self.is_running = False

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"

    @classmethod
    def from_default(cls,
        name: str,
        params: dict = {},
        defaults: dict = None
    ):
        default: dict = cls._default_fields.get(name)
        if default is None:
            raise DefaultNotFound(
                f"{name!r} is not a default Field name.")
        spec = {}
        spec.update(default)
        spec.update(params)
        return cls(**spec)

    def get_icon(self, default=None):
        if self.bar is None:
            return default
        icon = self.term_icon if self.bar.stream.isatty() else self.gui_icon
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
        bar = self.bar
        running = bar._can_run
        run_once = self.run_once
        override_queue = bar._override_queue
        overrides_refresh = self.overrides_refresh
        # self.is_running = True

        field_name = self.name
        field_buffers = bar._buffers
        interval = self.interval
        func = self._callback
        args = self.args
        kwargs = self.kwargs

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

                    # print(f"{(self.bar._loop == asyncio.get_running_loop()) = }")
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
                    assert self.bar._loop.is_running()
                    pass

            if run_once:
                break

            await asyncio.sleep(interval)

    def run_threaded(self):
        '''Run a blocking function in a thread
        and send its updates to the bar.'''
        self._check_bar()
        bar = self.bar
        running = bar._can_run
        run_once = self.run_once
        override_queue = bar._override_queue
        overrides_refresh = self.overrides_refresh
        # self.is_running = True

        field_name = self.name
        field_buffers = bar._buffers
        interval = self.interval
        func = self._callback
        args = self.args
        kwargs = self.kwargs

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
        self.thread = threading.Thread(
            target=self.run_threaded,
            name=self.name
        )
        self.thread.start()

    def _check_bar(self):
        '''Raises MissingBar if self.bar is None.'''
        if self.bar is None:
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
        'battery': Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=True),
        'net_stats': Field(name='net_stats', func=get_net_stats, interval=10),
        'datetime': Field(name='datetime', func=get_datetime, interval=1/8),
    }

    def __init__(
        self,
        /,
        fields: Iterable[Field|str] = None,
        fmt: str = None,
        sep: str = '|',
        refresh: float = 1.0,
        stream: IO = sys.stdout,
        show_empty: bool = False,
        field_params: dict = None,
        gui_sep: str = None,
        term_sep: str = None,
        override_cooldown: float = 1/60,
        thread_cooldown: float = 1/8
    ):

        io_meths = ('write', 'flush', 'isatty')
        if not all(hasattr(stream, a) for a in io_meths):
            _io_meths = [a + '()' for a in io_meths]
            raise InvalidOutputStream(
                f"Output stream {stream!r} needs "
                f"{join_options(_io_meths, final=' and ')} methods.")
        self.stream = stream

        if fmt is None:
            if fields is None:
                raise ValueError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required.")

            if not hasattr(fields, '__iter__'):
                raise ValueError("The fields argument must be iterable.")

            if all(s is None for s in (gui_sep, term_sep, sep)):
                raise UndefinedSeparator(
                    "A separator is required when fmt is None.")

            field_names = [
                getattr(f, 'name') if isinstance(f, Field)
                else f
                for f in fields
            ]

        elif not isinstance(fmt, str):
            raise TypeError(
                f"Format string 'fmt' must be a string, not {type(fmt)}")

        else:
            try:
                # Parse the field names found within fmt only if it is defined.
                # They may be used later to join field buffers for printing.
                field_names = [
                    name
                    for m in Formatter().parse(fmt)
                    if (name := m[1]) is not None
                ]
            except ValueError:
                e = BadFormatString(f"Invalid bar format string: {fmt}")
                raise e from None

            if '' in field_names:
                raise BadFormatString(
                    "The bar's format string contains positional fields: '{}'")

            #TODO: Try getting fields from the default field dict.
            # fields = [self._default_fields.get(fname) for fname in field_names]
            fields = list(field_names)
        self.field_names = field_names
        self.fmt = fmt
        self.term_sep = term_sep
        self.gui_sep = gui_sep
        self.separator = self.get_separator(sep)
        self.default_sep = sep

        default_field_params = dict(args=[], kwargs={})
        if field_params is None:
            field_params = {}

        self._fields = {}
        self._buffers = {}
        self._ordering = []

        for field in fields:
            if isinstance(field, str):
                # Try getting the field func from the default function dict.
                #TODO: Also try getting the entire field from _default_fields!
                field_func = self._field_funcs.get(field)
                if field_func is None:
                    raise InvalidField(f"Unrecognized field name: {field!r}")

                if field in self._ordering:
                    self._ordering.append(field)
                    continue

                params = field_params.get(field, default_field_params)

                field = Field(
                    name=field,
                    func=field_func,
                    bar=self,
                    **params
                )

            elif isinstance(field, Field):
                field.bar = self
                self._ordering.append(field.name)

            else:
                raise InvalidField(f"Invalid field: {field}")

            self._fields[field.name] = field
            self._buffers[field.name] = ''
            # self._ordering.append(field.name)

        # Whether empty fields are shown with the rest when fmt is None:
        self.show_empty_fields = show_empty

        # Set the bar's normal refresh rate, that is, how often it is printed:
        self.refresh_rate = refresh

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
    def in_a_tty(self):
        if self.stream is None:
            return False
        return self.stream.isatty()

    @property
    def clearline_char(self):
        if self.stream is None:
            return None
        clearline = CLEAR_LINE if self.stream.isatty() else ''
        return clearline

    def get_separator(self, default=None):
        if self.stream is None:
            return default
        sep = self.term_sep if self.stream.isatty() else self.gui_sep
        if sep is None:
            sep = default
        return sep

    def run(self, stream: IO = None):
        '''Run the bar. Block until an exception is raised and exit smoothly.'''
        if stream is not None:
            self.stream = stream

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
                self.stream.write('\n')
                self.stream.write(CSI + UNHIDE_CURSOR)
            self._shutdown()

    async def _startup(self):
        '''Schedule field coroutines, threads and the line printer to be run
        in parallel.'''
        field_coros = []
        for field in self._fields.values():
            # Do not run fields which have a constant output;
            # only set the bar buffer.
            if field.constant_output is not None:
                self._buffers[field.name] = field.constant_output
                continue

            if field.is_threaded:
                await field.send_to_thread()
            else:
                field_coros.append((field.run()))

        await asyncio.gather(
            *field_coros,
            # self._schedule_threads(),
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
            if field.is_threaded and field.thread is not None:
##                # Optionally kill threads that are still blocking program exit.
##                if self._kill_threads:
##                    signal.pthread_kill(field.thread.ident, self._kill_signal)
##                    print(f"Sent {self._kill_signal} to {field.name}")
                field.thread.join()
                # print(f"{field.thread.name}: {field.thread.is_alive() = }")

    async def _schedule_threads(self):
        '''Sends fields to threads if they are meant to be threaded.'''
        for field in self._fields.values():
            if field.is_threaded and field.constant_output is None:
                await field.send_to_thread()

    async def _continuous_line_printer(self, end: str = '\r'):
        '''The bar's primary line-printing mechanism.
        Fields are responsible for sending updates to the bar's buffers.
        This only writes using the current buffer contents.'''
        use_format_str = (self.fmt is not None)
        stream = self.stream
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
                    for field in self._ordering
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
        stream = self.stream
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
                        # for field in self.field_names
                        for field in self._ordering
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

        # self.read_file()
        if config_file is None:
            config_file = os.path.expanduser('~/bar.conf')

        config_file = 'bar.conf'
        self.file = config_file
        with open(config_file, 'r') as f:
            raw_data = f.read()

        self.yaml = strictyaml.load(raw_data)
        # self.data = self.yaml.data
        self.fields = self.get_fields()

        parms = self.yaml.copy()
        self.bar_params = parms.data
        self.bar_params.pop('fields')
        try:
            self.bar_params.pop('field_params', None)
            self.bar_params.pop('stream', None)
        except KeyError:
            pass

        bools = {}
        bools['show_empty'] = parms.get('show_empty')
        bools['override_cooldown'] = parms.get('override_cooldown')
        self.bar_params.update(
            self._cast_elements(bools, str_to_bool, "a boolean")
        )

        floats = {}
        floats['refresh'] = parms.get('refresh')
        floats['thread_cooldown'] = parms.get('thread_cooldown')
        floats['override_cooldown'] = parms.get('override_cooldown')
        self.bar_params.update(self._cast_elements(floats, float, "a number"))

        # updates = {}
        # print(floats)

    def _make_error_message(self,
        label: str,
        blame = None,
        expected: str = None,
        file: str = None,
        line: int = None,
        indent: str = "  ",
        details: Iterable[str] = None
    ):
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

    def _cast_elements(self,
        vals: dict,
        typ,
        expect: str,
        keys: Iterable[str] = None,
        # follow=None,
        **kwargs
    ):
        # This happens when
        # (When does this happen?)
        if not isinstance(vals, dict):
            text = ''
            follow = kwargs.get('follow')
            if follow is not None:
                text = follow.lines().splitlines()[0]
            raise ConfigInvalidField(repr(text))

        ret = {}
        for name, val in vals.items():
            if val is not None:
                try:
                    if keys is not None:
                        if name in keys:
                            ret[name] = typ(val)
                    else:
                        ret[name] = typ(val)
                except ValueError:
                    raise ConfigCannotConvert(expect, val)

        return ret

    def _validate_yaml(self):
        if self.yaml.get('fields') is None:
            raise ConfigInvalid(
                f"The config file {self.file!r} is missing a 'fields' array.")
        if not self.yaml['fields'].is_sequence():
            raise ConfigInvalid(
                f"'fields' in config file {self.file!r} must be an array of "
                f"valid field names and or nonempty dictionaries."
            )

    def write_config(self):
        file = os.path.expanduser('~/bar.conf')
        field_names = list(Field._default_fields.keys())
        yaml = strictyaml.as_document(field_names).as_yaml()
        with open(file, 'w') as f:
            f.write(yaml)

    def get_fields(self):
        self._validate_yaml()
        to_float = ('interval',)
        to_bool = ('override_refresh_rate', 'run_once', 'threaded')

        fields = []
        for field in self.yaml['fields']:
            # print(field.data)
            if isinstance(field.data, dict):
                try:
                    ((name, params),) = field.data.items()
                    start_line = field[name].start_line

                    floats = self._cast_elements(
                        keys=to_float,
                        vals=params,
                        typ=float,
                        expect="a number",
                        follow=field[name]
                    )

                    bools = self._cast_elements(
                        keys=to_bool,
                        vals=params,
                        typ=str_to_bool,
                        expect="a boolean",
                        follow=field[name]
                    )

                    params.update(**floats, **bools)
                    fields.append(Field.from_default(name, params))

                except ConfigCannotConvert as e:
                    err = self._make_error_message(
                        label=f"field option {name!r}",
                        blame=repr(e.args[1]),
                        expected=e.args[0],
                        file=self.file,
                        line=start_line
                    )
                    raise ConfigCannotConvert('\n' + err) from e

                except ConfigInvalidField as e:
                    err = self._make_error_message(

                        label=f"field {name!r}",
                        # blame=params,
                        blame=e.args[0], #.lines(),
                        expected=(
                            "the name of a default field or a dictionary "
                            "with optional overrides to a default field"
                        ),
                        file=self.file,
                        line=start_line,
                        details=(
                            f"Invalid field definition block.",
                            # f"'fields' must be an array of valid "
                            # f"field names and or nonempty dictionaries."
                        )
                    )

                    raise ConfigInvalidField('\n' + err)

                # An error about the dictionary update sequence means
                # something other than a dictionary or field name was passed.
                except ValueError:
                    print("Got ValueError!")
                    err = self._make_error_message(
                        file=self.file,
                        line=start_line,
                        label=f"field {name!r}",
                        blame=params,
                        expected="a dictionary",
                        details=("Invalid field definition.",),
                    )
                    raise ConfigInvalid('\n' + err)

                # Invalid keyword arguments are the cause of TypeErrors.
                except TypeError as e:
                    err = self._make_error_message(
                        file=self.file,
                        line=start_line,
                        label=f"field {name!r}",
                        blame=', '.join(repr(k) for k in params.keys()),
                        details=(
                            # f"Invalid keyword arguments for Field.__init__():",
                            f"At least one of the following keyword arguments "
                            f"for Field.__init__() are invalid:",
                        )
                    )
                    raise ConfigInvalid('\n' + err) from e

            elif isinstance(field.data, str):
                fields.append(Field.from_default(field.data))

            else:

                err = self._make_error_message(
                    file=self.file,
                    line=field.start_line,
                    label=f"field {field.data!r}",
                    expected="a dictionary",
                    details = (
                        f"Invalid field definition block.",
                        f"'fields' must be an array of valid "
                        f"field names and or nonempty dictionaries."
                    )
                )

                raise ConfigInvalid('\n' + err) from None

        return fields

    def get_bar(self):
        params = self.bar_params.copy()
        params.update(fields=self.fields)
        return Bar(**params)

def main():
    fhostname = Field(name='hostname', func=get_hostname, run_once=True, term_icon='')
    fuptime = Field(name='uptime', func=get_uptime, kwargs={'fmt': '%-jd:%-Hh:%-Mm'}, term_icon='Up:')
    fcpupct = Field(name='cpu_usage', func=get_cpu_usage, interval=2, threaded=True, term_icon='CPU:')
    fcputemp = Field(name='cpu_temp', func=get_cpu_temp, interval=2, threaded=True, term_icon='')
    fmem = Field(name='mem_usage', func=get_mem_usage, interval=2, term_icon='Mem:')
    fdisk = Field(name='disk_usage', func=get_disk_usage, interval=2, term_icon='/:')
    # fbatt = Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=False, term_icon='Bat:')
    fbatt = Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=True, term_icon='Bat:')
    fnet = Field(name='net_stats', func=get_net_stats, interval=4, term_icon='', threaded=False)

    fdate = Field(
        # fmt="<3 {icon}{}!",
        name='datetime',
        func=get_datetime,
        interval=1/8,
        override_refresh_rate=True,
        term_icon='?'
    )
    fcounter = Field(name='counter', func=counter, interval=1, args=[COUNT])

    # fdate = Field(name='datetime', func=get_datetime, interval=1, override_refresh_rate=False, term_icon='')
    # fdate = Field(func=get_datetime, interval=0.5, override_refresh_rate=False)
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

    # fields = [Field(name='test', interval=1/16, func=(lambda args=None, kwargs=None: "WORKING"), override_refresh_rate=True, term_icon='&', gui_icon='@')]

    global bar
    # bar = Bar(fields=fields)
    # bar = Bar(fields=fields, sep=None, fmt=None)  # Raise UndefinedSeparator

    # fmt = "Up{uptime} | CPU: {cpu_usage}, {cpu_temp}|Disk: {disk_usage} Date:{datetime}..."
    # fmt = None
    # bar = Bar(fmt=fmt)

    # bar = Bar(fields=[fdate, fcount, fhostname, fuptime])

    CFG = Config('bar.conf')
    bar = CFG.get_bar()
    bar.run()


if __name__ == '__main__':
    main()

