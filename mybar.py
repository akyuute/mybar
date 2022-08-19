#!/usr/bin/python

import os
import sys
import time
import psutil
import signal
import asyncio
import inspect
import threading
from string import Formatter
from typing import Iterable, Callable, IO

from field_funcs import *


CSI = '\033['  # Unix terminal escape code (control sequence introducer)
CLEAR_LINE = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR = '?25l'
UNHIDE_CURSOR = '?25h'


#TODO: Field icons and fmt!
#TODO: Finish Mocp line!
#TODO: Implement killing threads!
#TODO: Implement align_to_clock!
#TODO: Implement dynamic icons!


class InvalidOutputStream(Exception):
    pass
class UndefinedSeparator(Exception):
    pass
class BadFormatString(Exception):
    pass
class InvalidField(Exception):
    pass
class MissingBar(Exception):
    pass
class UndefinedIcon(Exception):
    pass


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


class Field:
    def __init__(self,
        /,
        name: str = None,
        func: Callable = None,
        icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        align_to_clock=False,
        override_refresh_rate=False,
        threaded=False,
        constant_output: str = None,
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

        self.bar = bar

        if fmt is None:
            if all(s is None for s in (gui_icon, term_icon, icon)):
                raise UndefinedIcon("An icon is required when fmt is None.")
        self.fmt = fmt
        self.term_icon = term_icon
        self.gui_icon = gui_icon
        self.icon = self.get_icon(icon)

        if inspect.iscoroutinefunction(self._func) or threaded:
            self._callback = self._func
        else:
            # Wrap a synchronous function call
            self._callback = self._asyncify

        if name is None:
            name = self._func.__name__
        self.name = name

        self.align_to_clock = align_to_clock
        self.buffer = None
        self.constant_output = constant_output
        self.interval = interval
        self.is_threaded = threaded
        self.overrides_refresh = override_refresh_rate
        self.thread = None
        # self.is_running = False

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

    async def _send_contents(self, field_name, updates: str):
        '''Send new field contents to the bar.'''
        if self.fmt is None:
            contents = updates
        else:
            contents = self.fmt.format(updates)
        self.bar._buffers[field_name] = contents

    async def _send_and_override(self, field_name, updates: str):
        '''Send new field contents to the bar's override queue
        and print a new line between refresh cycles.'''
        if self.fmt is None:
            contents = updates
        else:
            contents = self.fmt.format(updates)
        self.bar._buffers[field_name] = contents
        try:
            self.bar._override_queue.put_nowait((field_name, contents))
        except asyncio.QueueFull:
            # Since the bar buffer was just updated, the change is effective,
            # and it may still appear while the queue handles another override.
            # If not, the line will always update at the next refresh cycle.
            pass

    async def run(self):
    #async def run(self, func, interval, args, kwargs):
        '''Asynchronously run a non-threaded field's callback
        and send updates to the bar.'''
        if self.bar is None:
            raise MissingBar("Fields cannot run until they belong to a Bar.")
        bar = self.bar
        icon = self.icon

        # self.is_running = True

        func = self._callback
        field_name = self.name
        last_val = None
        interval = self.interval
        args = self.args
        kwargs = self.kwargs

        if self.overrides_refresh:
            on_update = self._send_and_override
        else:
            on_update = self._send_contents

        while not bar._stopped.is_set():
            res = await func(*args, **kwargs)

            if res != last_val:
                last_val = res
                await on_update(field_name, updates=res)

            await asyncio.sleep(interval)

    # def run_threaded(self, timeout: float = 1/30):
    def run_threaded(self, timeout: float = 1/8):
    #def run_threaded(self, func, interval, args, kwargs):
        '''Run a blocking function in a thread
        and send its updates to the bar.'''
        if self.bar is None:
            raise ValueError("Fields cannot be run until they are part of a Bar.")
        bar = self.bar
        # icon = self.term_icon if bar.stream.isatty() else self.gui_icon
        # icon = self.get_icon()
        icon = self.icon

        # self.is_running = True

        func = self._callback
        field_name = self.name
        last_val = None
        interval = self.interval
        args = self.args
        kwargs = self.kwargs

        if self.overrides_refresh:
            on_update = (self._send_and_override)
        else:
            on_update = (self._send_contents)

        # An event loop is needed to run either update function.
        loop = asyncio.new_event_loop()
        # If the field's callback is asynchronous, run it in the event loop.
        is_async = inspect.iscoroutinefunction(func)

        count = 0
        while not bar._stopped.is_set():

            # Instead of blocking the entire interval, use a quicker timeout.
            # A shorter timeout means more chances to check if the bar stops.
            # A thread, then, cancels timeout seconds after its function returns
            # rather than interval seconds.
            if count != int(interval / timeout):
                count += 1
                time.sleep(timeout)
                continue

            if is_async:
                res = loop.run_until_complete(func(*args, **kwargs))
            else:
                res = func(*args, **kwargs)

            if res != last_val:
                last_val = res
                loop.run_until_complete(on_update(field_name, updates=res))

            count = 0

        loop.stop()
        loop.close()

    async def send_to_thread(self):
        '''Make and start a thread in which to run the field's callback.'''
        self.thread = threading.Thread(
            target=self.run_threaded,
            name=self.name,
        )
        self.thread.start()


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

        default_field_params = dict(args=[], kwargs={})
        if field_params is None:
            field_params = {}

        self._fields = {}
        self._buffers = {}

        for field in fields:
            if isinstance(field, str):
                self._buffers[field] = ''

                # Try getting the field func from the default function dict.
                #TODO: Also try getting the entire field from _default_fields!
                field_func = self._field_funcs.get(field)
                if field_func is None:
                    raise InvalidField(f"Unrecognized field name: {field!r}")

                params = field_params.get(field, default_field_params)

                field = Field(
                    name=field,
                    func=field_func,
                    bar=self,
                    **params
                )

            elif isinstance(field, Field):
                field.bar = self

            else:
                raise InvalidField(f"Invalid field: {field}")

            self._fields[field.name] = field
            self._buffers[field.name] = ''

        # Make a queue to which fields with overrides_refresh send updates.
        # Only one override is processed per refresh cycle to reduce the chance
        # of flickering in a GUI.
        self._override_queue = asyncio.Queue(maxsize=1)

        self.refresh_rate = refresh
        # Whether empty fields are joined when fmt is None:
        self.show_empty_fields = show_empty
        # Setting this Event cancels all fields:
        self._stopped = threading.Event()
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

##        if use_format_str:
##            line_maker = self._make_line_from_fmt
##        else:
##            line_maker = self._make_line_from_list

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        stream.flush()
        start_time = time.monotonic_ns()
        while not self._stopped.is_set():

            if use_format_str:
                line = self.fmt.format_map(self._buffers)
            else:
                line = sep.join(
                    buf
                    for field in self.field_names
                        if (buf := self._buffers[field])
                        or show_empty_fields
                )

            # line = await line_maker()

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

        if self.in_a_tty:
            beginning = clearline + end
        else:
            beginning = clearline

##        if use_format_str:
##            line_maker = self._make_line_from_fmt
##        else:
##            line_maker = self._make_line_from_list

        try:
            while not self._stopped.is_set():

                field, contents = await self._override_queue.get()

                if use_format_str:
                    line = self.fmt.format_map(self._buffers)
                else:
                    line = sep.join(
                        buf
                        for field in self.field_names
                            if (buf := self._buffers[field])
                            or show_empty_fields
                    )

                # line = await line_maker()

                stream.write(beginning + line + end)
                stream.flush()

        # SIGINT raises RuntimeError saying an event loop is closed.
        except RuntimeError:
            pass

    async def _startup(self):
        '''Schedule field coroutines, threads and the line printer to be run
        in parallel.'''
        field_coros = []
        for field in self._fields.values():
            if field.constant_output is not None:
                # Do not run fields which have a constant output;
                # only set the bar buffer.
                self._buffers[field.name] = field.constant_output
                continue
            if not field.is_threaded:
                field_coros.append((field.run()))

        await asyncio.gather(
            *field_coros,
            self._schedule_threads(),
            self._check_queue(),
            self._continuous_line_printer(),
        )

    async def _schedule_threads(self):
        '''Sends fields to threads if they are meant to be threaded.'''
        for field in self._fields.values():
            if field.is_threaded and field.constant_output is None:
                await field.send_to_thread()

    def run(self, stream=None):
        '''Run the bar.
        Block until an exception is raised and exit smoothly.'''
        if stream is not None:
            self.stream = stream
        # self.separator = self.get_separator()
        if self.fmt is None and self.separator is None:
            raise UndefinedSeparator(
                f"No separator is defined for stream {self.stream}")

        try:
            self._loop.run_until_complete(self._startup())
            # asyncio.run(self._startup())

        except KeyboardInterrupt:
            pass

        finally:
            if self.in_a_tty:
                self.stream.write('\n')
                self.stream.write(CSI + UNHIDE_CURSOR)
            self._shutdown()

    def _shutdown(self):
        '''Set the bar's stop event and join threads.'''
        self._stopped.set()
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

    async def _make_line_from_list(self):
        '''An async callback for making a line by joining field buffers.'''
        line = sep.join(
            buf
            for field in self.field_names
                if (buf := self._buffers[field])
                or self.show_empty_fields
        )
        return line

    async def _make_line_from_fmt(self):
        '''An async callback for making a line from a format string.'''
        return self.fmt.format_map(self._buffers)

def main():
    fhostname = Field(name='hostname', func=get_hostname, interval=10, term_icon='')
    fuptime = Field(name='uptime', func=get_uptime, kwargs={'fmt': '%-jd:%-Hh:%-Mm'}, term_icon='Up:')
    fcpupct = Field(name='cpu_usage', func=get_cpu_usage, interval=2, threaded=True, term_icon='CPU:')
    fcputemp = Field(name='cpu_temp', func=get_cpu_temp, interval=2, threaded=True, term_icon='')
    fmem = Field(name='mem_usage', func=get_mem_usage, interval=2, term_icon='Mem:')
    fdisk = Field(name='disk_usage', func=get_disk_usage, interval=2, term_icon='/:')
    # fbatt = Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=False, term_icon='Bat:')
    fbatt = Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=True, term_icon='Bat:')
    fnet = Field(name='net_stats', func=get_net_stats, interval=4, term_icon='', threaded=False)


    fdate = Field(fmt="<3 {}!", name='datetime', func=get_datetime, interval=1/8, override_refresh_rate=True, term_icon='')
    # fdate = Field(name='datetime', func=get_datetime, interval=1, override_refresh_rate=False, term_icon='')
    # fdate = Field(func=get_datetime, interval=0.5, override_refresh_rate=False)
    # fdate = Field(func=get_datetime, interval=1)
    # mocpline = Field(name='mocpline', func=

    global fields
    fields = (
        Field(name='isatty', func=(lambda args=None, kwargs=None: str(bar.in_a_tty)), interval=9)
        ,fhostname,
        fuptime,
        fcpupct,
        fcputemp,
        fmem,
        fdisk,
        fbatt,
        fnet,
        fdate,
    )

    # fields = [Field(name='test', interval=1/16, func=(lambda args=None, kwargs=None: "WORKING"), override_refresh_rate=True, term_icon='&', gui_icon='@')]

    global bar
    bar = Bar(fields=fields)

    # fields = 
    # fmt = "Up{uptime} | CPU: {cpu_usage}, {cpu_temp}|Disk: {disk_usage} Date:{datetime}{..." #{count}"
    # fmt = None
    # bar = Bar(fmt=fmt, fields=fields, refresh=1)
    # bar = Bar(fmt=fmt)

    # bar = Bar(fields=[fdate, fcount, fhostname, fuptime])
    # bar = Bar(fields='datetime hostname counter hostname uptime hostname'.split(), field_params={'counter': dict(args=(COUNT,), interval=0.333)})

    bar.run()

if __name__ == '__main__':
    main()

