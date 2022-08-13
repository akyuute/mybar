#!/usr/bin/python

import os
import sys
import time
import psutil
import signal
import asyncio
import inspect
import threading
from typing import Callable
from string import Formatter

from field_funcs import *


CSI = '\033['  # Unix terminal escape code (control sequence introducer)
CLEAR_LINE = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR = '?25l'
UNHIDE_CURSOR = '?25h'


#TODO: Re-implement actual buffer joins!
#TODO: Implement killing threads!
#TODO: Implement align_to_clock!
#TODO: Implement dynamic icons!


class InvalidField(Exception):
    pass
class BadFormatString(Exception):
    pass
class MissingBar(Exception):
    pass


class Field:
    def __init__(self,
        /,
        name: str = None,
        func: Callable = None,
        term_icon: str = '',
        gui_icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        align_to_clock=False,
        override_refresh_rate=False,
        threaded=False,
        constant_output: str = None,
        bar=None,
        args = [],
        kwargs = {}
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

        if inspect.iscoroutinefunction(self._func) or threaded:
            self._callback = self._func
        else:
            # Wrap a synchronous function call
            self._callback = self._asyncify

        if name is None:
            name = self._func.__name__
        self.name = name

        self.term_icon = term_icon
        self.gui_icon = gui_icon

        if fmt is None:
            fmt = '{}'
        self.fmt = fmt

        self.align_to_clock = align_to_clock
        self.bar = bar
        self.buffer = None
        self.constant_output = constant_output
        self.icon = None
        self.interval = interval
        self.is_threaded = threaded
        self.overrides_refresh = override_refresh_rate
        self.thread = None
        self._stop_event = threading.Event()
        # self.is_running = False

##    def _is_running(self):
##        return not self._stop_event.is_set()

    async def _asyncify(self, *args, **kwargs):
        '''Wrap a synchronous function in a coroutine for simplicity.'''
        return self._func(*args, **kwargs)

    async def _send_formatted(self, field_name, updates: str):
        '''Format updated data before sending it to the bar buffers.'''
        self.bar._buffers[field_name] = self.fmt.format(updates)

    async def run(self):
    #async def run(self, func, interval, args, kwargs):
        '''Asynchronously run a non-threaded field's callback
        and send updates to the bar.'''
        if self.bar is None:
            raise MissingBar("Fields cannot run until they belong to a Bar.")
        bar = self.bar
        icon = self.term_icon if bar.stream.isatty() else self.gui_icon

        # self.is_running = True

        func = self._callback
        field_name = self.name
        fmt = icon + self.fmt
        buffer = self.buffer
        interval = self.interval
        args = self.args
        kwargs = self.kwargs

        if self.overrides_refresh:
            if bar.fmt is None:
                on_update = bar._print_line_from_list
            else:
                on_update = bar._print_line_from_fmt
        else:
            if self.fmt is None:
                on_update = bar._take_updates
            else:
                on_update = self._send_formatted
        # print(f"{field_name}: {on_update.__name__}")

        # start = time.monotonic_ns()
        while not self._stop_event.is_set():
            res = await func(*args, **kwargs)

            out = fmt.format(res)
            if out != buffer:
                await on_update(field_name, updates=out)

            # if self.align_to_clock:
                # Syncing the sleep timer to the system clock prevents drifting
                # delay = (self.refresh_rate - (time.monotonic_ns() - start) % self.refresh_rate)
                # await asyncio.sleep(timeout)
            await asyncio.sleep(interval)

        # Syncing the sleep timer to the system clock prevents drifting
        # time.sleep(refresh_rate - (time.time() - start) % refresh_rate)

    # def run_threaded(self, timeout: float = 1/30):
    def run_threaded(self, timeout: float = 1/8):
    #def run_threaded(self, func, interval, args, kwargs):
        '''Run a blocking function and send
        '''
        if self.bar is None:
            raise ValueError("Fields cannot be run until they are part of a Bar.")
        bar = self.bar
        icon = self.term_icon if bar.stream.isatty() else self.gui_icon

        # self.is_running = True

        func = self._callback
        field_name = self.name
        fmt = icon + self.fmt
        interval = self.interval
        args = self.args
        kwargs = self.kwargs

        # If the field's callback is asynchronous,
        # make an event loop in which to run it.
        is_async = inspect.iscoroutinefunction(func)
        loop = asyncio.new_event_loop()

        if self.overrides_refresh:
            if bar.fmt is not None:
                on_update = bar._print_line_from_fmt
            else:
                on_update = bar._print_line_from_list
        else:
            on_update = bar._take_updates
        # print(f"{field_name}: {on_update.__name__}")

        count = 0
        while not self._stop_event.is_set():
            if count != int(interval / timeout):
                count += 1
                time.sleep(timeout)
                continue

            if is_async:
                res = loop.run_until_complete(func(*args, **kwargs))
            else:
                res = func(*args, **kwargs)

            out = fmt.format(res)
            if out != self.buffer:
                # The bar's printer and updater functions are always asynchronous,
                # so they need to be run by the new thread's event loop.
                loop.run_until_complete(on_update(field_name, updates=out))

            count = 0
        # loop.stop()
        # loop.close()

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
        'battery': Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=False),
        'net_stats': Field(name='net_stats', func=get_net_stats, interval=10),
        'datetime': Field(name='datetime', func=get_datetime, interval=1/16, override_refresh_rate=True),
    # fdate = Field(func=get_datetime, interval=0.5, override_refresh_rate=False),
    }

    def __init__(
        self,
        /,
        fmt: str = None,
        fields=None,
        # sep: str = ' | ',
        term_sep: str = '|',
        gui_sep: str = ' | ',
        refresh: float = 1/16,
        field_params: dict = None,
        stream=None
    ):

        if fmt is None:
            if fields is None:
                raise ValueError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required.")

            if not hasattr(fields, '__iter__'):
                raise ValueError("The fields argument must be iterable.")

            if stream is None:
                stream = sys.stdout
            self.stream = stream

            sep = term_sep if self.stream.isatty() else gui_sep
            # if sep is None:
                # raise ValueError("'sep' is required when fmt is None.")

            field_names = [
                getattr(f, 'name') if isinstance(f, Field)
                else f
                for f in fields
            ]

            # Make a default format string pre-populated with the separator.
            #NOTE: This leaves the possibility of empty fields!
            fmt = '{' + ('}'+sep+'{').join(field_names) + '}'

        elif not isinstance(fmt, str):
            raise TypeError(
                f"Format string 'fmt' must be a string, not {type(fmt)}")

        else:
            try:
                # When only fmt is set, parse the field names found within it.
                # They are used later to print lines by joining field buffers.
                field_names = [
                    name
                    for m in Formatter().parse(fmt)
                    if (name := m[1]) is not None
                ]
            except ValueError:
                raise BadFormatString("Your bar's format string sucks.") from None

            if '' in field_names:
                raise BadFormatString(
                    "The bar's format string contains positional fields: '{}'")
            # fields = [self._default_fields.get(fname) for fname in field_names]
            fields = list(field_names)

        self.field_names = field_names
        self.fmt = fmt
        self.refresh_rate = refresh
        self.separator = sep
        self._stop_event = threading.Event()
        self._loop = asyncio.new_event_loop()

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
                    **params
                )

            elif not isinstance(field, Field):
                raise InvalidField(f"Invalid field: {field}")

            field.bar = self
            self._fields[field.name] = field
            self._buffers[field.name] = ''

    async def _continuous_line_printer(self, end: str = '\r'):
        '''The bar's primary line-printing mechanism.
        Fields are responsible for sending updates to the bar's buffers.
        This only writes using the current buffer contents.'''
        use_format_str = (self.fmt is not None)

        while not self._stop_event.is_set():
            if use_format_str:
                # print(f"{self._buffers = }")
                # printer = self._print_line_from_fmt
                # print(repr(self.fmt))
                line = self.fmt.format_map(self._buffers)
                # print(f"{line = }")
            else:
                # printer = self._print_line_from_list
                # line = self.sep.join(self._fields_list)
                # line = self.sep.join(self._fields.values())
                line = self.separator.join(
                    self._buffers[field]
                    for field in self.field_names
                )

            self.stream.write(CLEAR_LINE + end + line + end)
            # await printer()
            await asyncio.sleep(self.refresh_rate)

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
            self._continuous_line_printer()
        )

    async def _schedule_threads(self):
        '''Send fields to threads if they are meant to be threaded.'''
        for field in self._fields.values():
            if field.is_threaded and field.constant_output is None:
                await field.send_to_thread()

    def run(self, stream=None):
        '''Run the bar.
        Block until an exception is raised and exit smoothly.'''
        if stream is None:
            stream = self.stream

        try:
            stream.write(CSI + HIDE_CURSOR)
            self._loop.run_until_complete(self._startup())

        except KeyboardInterrupt:
            pass

        finally:
            stream.write('\n')
            stream.write(CSI + UNHIDE_CURSOR)
            self._shutdown()

    def _shutdown(self):
        '''Set stop events for the bar and fields and join threads.'''
        # Indicate that the bar has stopped, for the sake of being thorough.
        self._stop_event.set()
        # Cancel all threads and coroutines at the next iteration.
        for field in self._fields.values():
            field._stop_event.set()
            if field.is_threaded and field.thread is not None:
##                # Optionally kill threads that are still blocking program exit.
##                if self._kill_threads:
##                    signal.pthread_kill(field.thread.ident, self._kill_signal)
##                    print(f"Sent {self._kill_signal} to {field.name}")
                field.thread.join()
                # print(f"{field.thread.name}: {field.thread.is_alive() = }")

    async def _take_updates(self, field_name: str, updates: str):
        '''Unless its override_refresh_rate is True or it has a fmt,
        a field uses this to update the buffer when it has new data.'''
        if updates is not None:
            self._buffers[field_name] = updates
 
    async def _print_line_from_list(self,
        field_name: str,
        updates: str = None,
        end: str = '\r',
    ):
        '''Print a new line made by joining field buffers with the separator
        when a field has new data and its override_refresh_rate is True.'''
        if updates is not None:
            self._buffers[field_name] = updates
        line = self.separator.join(self._buffers[field] for field in self.field_names)
        self.stream.write(CLEAR_LINE + line + end)

    async def _print_line_from_fmt(self,
        field_name: str = None,
        updates: str = None,
        end: str = '\r',
    ):
        '''Print a new line using the bar's format string when a field has new
        data and its override_refresh_rate is True.'''
        if updates is not None:
            self._buffers[field_name] = updates
        line = self.fmt.format_map(self._buffers)
        self.stream.write(CLEAR_LINE + line + end)


def main():
    fhostname = Field(name='hostname', func=get_hostname, interval=10, term_icon='')
    fuptime = Field(name='uptime', func=get_uptime, kwargs={'fmt': '%-jd:%-Hh:%-Mm'}, term_icon='Up:')
    fcpupct = Field(name='cpu_usage', func=get_cpu_usage, interval=2, threaded=True, term_icon='CPU:')
    fcputemp = Field(name='cpu_temp', func=get_cpu_temp, interval=2, threaded=True, term_icon='')
    fmem = Field(name='mem_usage', func=get_mem_usage, interval=2, term_icon='Mem:')
    fdisk = Field(name='disk_usage', func=get_disk_usage, interval=10, term_icon='/:')
    fbatt = Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=False, term_icon='Bat:')
    fnet = Field(name='net_stats', func=get_net_stats, interval=10, term_icon='')


    fdate = Field(fmt="<3 {}!", name='datetime', func=get_datetime, interval=1/16, override_refresh_rate=True, term_icon='')
    # fdate = Field(func=get_datetime, interval=0.5, override_refresh_rate=False)
    # fdate = Field(func=get_datetime, interval=1)
    # mocpline = Field(name='mocpline', func=

    fields = (
        fhostname,
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
    bar = Bar(fields=fields, refresh=1, term_sep=' ')
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
    # pass

