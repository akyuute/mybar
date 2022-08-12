#!/usr/bin/python

import os
import sys
import time
import psutil
import signal
import asyncio
import inspect
import datetime
import threading
from typing import Callable
from string import Formatter
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from field_funcs import *

COUNT = [0]
#if os.name == 'posix':
CSI = '\033['
CLEAR_LINE = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR = '?25l'
UNHIDE_CURSOR = '?25h'


class InvalidField(Exception):
    pass
class MalformedString(Exception):
    pass


class Field:
    def __init__(self,
        /,
        name: str = None,
        func=None,
        icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        align_to_seconds=False,
        override_refresh_rate=False,
        threaded=False,
        constant_output: str = None,
        args = [],
        kwargs = {}
    ):
        if func is None and value is None:
            raise ValueError(
                f"Either a function that returns a string or "
                f"a constant value string is required."
            )

        # self.name = name

        if not callable(func):
            raise TypeError(
                f"Type of 'func' must be callable, "
                f"not {type(self._callback)}"
            )
        self._func = func

        if inspect.iscoroutinefunction(self._func): 
            self._callback = self._func
            # self.asyncfunc = self._func
        else:
            # self._callback = self._asyncify
            self._callback = self._func
            # self.asyncfunc = self._asyncify

        if name is None:
            name = self._func.__name__
        self.name = name

        self.icon = icon

        if fmt is None:
            fmt = self.icon + '{}'
        self.fmt = fmt

        self.constant_output = constant_output
        self.interval = interval
        self.align_to_seconds = align_to_seconds
        self.args = args
        self.kwargs = kwargs
        self.bar = None
        self.buffer = None
        self.overrides_refresh = override_refresh_rate
        self.is_threaded = threaded
        self.thread = None
        self._stop_event = threading.Event()
        # self.is_running = False

##    def _is_running(self):
##        return not self._stop_event.is_set()

    async def _asyncify(self, *args, **kwargs):
        '''Wrap a synchronous function in a coroutine for simplicity.'''
        return self._callback(*args, **kwargs)

    async def _send_updates(self, field_name, updates: str):
        '''Send new data to the bar buffers'''
        # bar = self.bar
        # if updates is not None:
        if self.fmt is None:
            self.bar._buffers[field_name] = updates
            # self.bar._buffers.update(updates)
        else:
            self.bar._buffers[field_name] = self.fmt.format(updates, )

 

    async def run(self):
    #async def run(self, func, interval, args, kwargs):
        '''Asynchronously run a non-threaded field's callback
        and send updates to the bar.'''
        # self.is_running = True
        func = self._callback
        field_name = self.name
        fmt = self.fmt
        # icon = self.icon
        buffer = self.buffer
        interval = self.interval
        bar = self.bar
        args = self.args
        kwargs = self.kwargs

        is_async = inspect.iscoroutinefunction(self._callback)
        func = self._callback if is_async else self._asyncify

        if self.overrides_refresh:
            if bar.fmt is None:
                on_update = bar._print_line_from_list
            else:
                on_update = bar._print_line_from_fmt
        else:
            if self.fmt is None:
                on_update = bar._take_updates
            else:
                on_update = self._send_updates
        # print(f"{field_name}: {on_update.__name__}")

        # start = time.monotonic_ns()
        while not self._stop_event.is_set():
            res = await func(*args, **kwargs)

            out = fmt.format(res)
            if out != buffer:
##                updates = {field_name: out}
                # updates = out
##                # print(repr(out))
                await on_update(field_name, updates=out)
            # if self.align_to_seconds:
                # Syncing the sleep timer to the system clock prevents drifting
                # delay = (self.refresh_rate - (time.monotonic_ns() - start) % self.refresh_rate)
                # await asyncio.sleep(timeout)
            await asyncio.sleep(interval)

        # Syncing the sleep timer to the system clock prevents drifting
        # time.sleep(refresh_rate - (time.time() - start) % refresh_rate)

    # def run_blocking(self, timeout: float = 1/30):
    def run_blocking(self, timeout: float = 1/8):
    #def run_blocking(self, func, interval, args, kwargs):
        # self.is_running = True

        func = self._callback
        field_name = self.name
        fmt = self.fmt
        # icon = self.icon
        interval = self.interval
        bar = self.bar
        args = self.args
        kwargs = self.kwargs

        loop = asyncio.new_event_loop()

        is_async = inspect.iscoroutinefunction(func)

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
            if count != interval / timeout:
                count += 1
                time.sleep(timeout)
                continue

            if is_async:
                res = loop.run_until_complete(func(*args, **kwargs))
            else:
                res = func(*args, **kwargs)

            out = fmt.format(res)
            if out != self.buffer:
                # print(ret)
                # updates = {field_name: icon + res}
                # updates = {field_name: out}
                # updates = out
                loop.run_until_complete(on_update(field_name, updates=out))
                # asyncio.run(bar._print_line(updates))
                # bar._loop.run_until_complete(send_updates)

            count = 0

    async def send_to_thread(self, ): #loop):
        self.thread = threading.Thread(
            target=self.run_blocking,
            name=self.name,
            args=self.args,
            kwargs=self.kwargs, 
        )
        # print(f"\nSending {self.name} to a thread...")
        self.thread.start()
        # print(f"Sent {self.name} to a thread.")


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
        *,
        fmt: str = None,
        fields=None,
        sep: str = ' | ',
        refresh: float = 1/16,
        field_params: dict = None
    ):

        '''
        $ mybar fields=1,2,3... sep='|' 
        $ mybar fmt="{foo} - {bar}"
        '''

        # if fields is None and fmt is None:
        if fmt is None:
            if fields is None:
                raise ValueError(
                    f"Either a list of Fields 'fields' "
                    f"or a format string 'fmt' is required.")

            #no fmt, has fields
            if not hasattr(fields, '__iter__'):
                raise ValueError("The fields argument must be iterable.")

            if sep is None:
                raise ValueError("'sep' is required when fmt is None.")

            field_names = [
                getattr(f, 'name')
                    if isinstance(f, Field)
                    else f
                for f in fields
            ]

            # Make a default format string pre-populated with the separator.
            # This leaves the possibility of empty fields.
            fmt = '{' + ('}'+sep+'{').join(field_names) + '}'

        #no fields, has fmt
        elif not isinstance(fmt, str):
            raise TypeError(f"Format string 'fmt' must be a string, not {type(fmt)}")

        #has fmt and is str; no fields
        else:
            try:
                # Used when printing lines by joining field buffers with the separator:
                field_names = [
                    name
                    for m in Formatter().parse(fmt)
                    if (name := m[1]) is not None
                ]
            except ValueError as e:
                raise MalformedString("Your bar's format string sucks.") from None
               # e.args =  

            if '' in field_names:
                raise ValueError("Format string contains positional fields: '{}'")
            # fields = [self._field_funcs.get(fname) for fname in field_names]
            fields = list(field_names)

        self.fmt = fmt
        self.separator = sep
        self.field_names = field_names
        self.refresh_rate = refresh
        self._stop_event = threading.Event()
        self._loop = asyncio.new_event_loop()
        # self._loop = asyncio.get_event_loop()
        # self._loop = NotImplementedError()
        self.last_line = ''

        default_field_params = dict(args=[], kwargs={})
        if field_params is None:
            field_params = {}

        self._fields = {}
        self._buffers = {}

        for field in fields:
            if isinstance(field, str):
                self._buffers[field] = ''

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

    async def _continuous_line_printer(self, stream=sys.stdout, end: str = '\r'):
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

            stream.write(CLEAR_LINE + line + end)
            # await printer()
            await asyncio.sleep(self.refresh_rate)

    async def _startup(self, ): #stream=sys.stdout):
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

    async def _schedule_threads(self): #loop):
        '''Send fields to threads if they are meant to be threaded.'''
        for field in self._fields.values():
            if field.is_threaded and field.constant_output is None:
                await field.send_to_thread()

    def run(self, stream=sys.stdout):
        '''Run the bar and block until an exception is raised.
        Exit smoothly.'''
        try:
            stream.write(CSI + HIDE_CURSOR)
            self._loop.run_until_complete(self._startup())
            # self._loop.run_until_complete(self._startup(stream))
        except KeyboardInterrupt:
            # print("\n")
            # print("KeyboardInterrupt caught in Bar.run()")
            # print("Shutting down...")
            pass

        finally:
            stream.write('\n')
            stream.write(CSI + UNHIDE_CURSOR)
            self._shutdown()
            # exit(0)

    def _shutdown(self):
        '''
        '''
        # Indicate that the bar has stopped, for the sake of completeness.
        self._stop_event.set()
        # Cancel all threads and coroutines at the next iteration.
        for field in self._fields.values():
            field._stop_event.set()
            if field.is_threaded and field.thread is not None:
##                print("\nSet _stop_event for", field.name)
##                if self._kill_threads:
##                    signal.pthread_kill(field.thread.ident, signal.SIGINT)
##                    print(f"Sent {self._kill_signal} to {field.name}")
                field.thread.join()
                # print(f"{field.thread.name}: {field.thread.is_alive() = }")
        # print("Joined threads.")

    # async def _take_updates(self, updates: dict):
    async def _take_updates(self, field_name, updates: str):
        '''Unless its override_refresh_rate is True,
        each field uses this to update the buffer when it has new data.'''
        if updates is not None:
            self._buffers[field_name] = updates

 
    async def _print_line_from_list(self,
        field_name: str,
        updates: dict = None,
        stream=sys.stdout,
        end: str = '\r',
    ):
        '''When the bar has no format string, fields whose override_refresh_rate
        is True use this to update the buffer and print a new line when there
        is new data.'''
        if updates is not None:
            self._buffers[field_name] = updates
        line = self.separator.join(self._buffers[field] for field in self.field_names)
        stream.write(CLEAR_LINE + line + end)

    async def _print_line_from_fmt(self,
        field_name: str = None,
        updates: str = None,
        # fmt: str = None,
        stream=sys.stdout,
        end: str = '\r',
    ):
        '''When the bar has a format string, fields whose override_refresh_rate
        is True use this to update the buffer and print a new line when there
        is new data.'''
        if updates is not None:
            self._buffers[field_name] = updates
        line = self.fmt.format_map(self._buffers)
        stream.write(CLEAR_LINE + line + end)

def main():
    fcount = Field(name='count', func=counter, interval=0.333, args=(COUNT,), override_refresh_rate=False)
    fhostname = Field(name='hostname', func=get_hostname, interval=10)
    fuptime = Field(name='uptime', func=get_uptime, kwargs={'fmt': '%-jd:%-Hh:%-Mm'})
    fcpupct = Field(name='cpu_usage', func=get_cpu_usage, interval=2, threaded=True, ) #kwargs={'interval': 2}, threaded=True)
    fcputemp = Field(name='cpu_temp', func=get_cpu_temp, interval=2, threaded=True)
    fmem = Field(name='mem_usage', func=get_mem_usage, interval=2)
    fdisk = Field(name='disk_usage', func=get_disk_usage, interval=10)
    fbatt = Field(name='battery', func=get_battery_info, interval=1, override_refresh_rate=False)
    fnet = Field(name='net_stats', func=get_net_stats, interval=10)


    fdate = Field(fmt="<3 {}!", name='datetime', func=get_datetime, interval=1/16, override_refresh_rate=True)
    # fdate = Field(func=get_datetime, interval=0.5, override_refresh_rate=False)

    # fdate = Field(func=get_datetime, interval=1)


    # mocpline = Field(name='mocpline', func=

    fields = (
        # fcount,
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

    global bar
    bar = Bar(fields=fields, refresh=1)
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

