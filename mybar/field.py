#TODO: NewType()! TypedDict()!
#TODO: collections.defaultdict, dict.fromkeys!
#TODO: Finish Mocp line!
#TODO: Implement dynamic icons!


__all__ = (
    'Field',
)


import asyncio
import threading
import time

from mybar import DEBUG
from mybar import field_funcs
from mybar import setups
from mybar.errors import *
from mybar.utils import (
    join_options,
    make_error_message,
)

from collections.abc import Callable, Sequence
from typing import NoReturn, TypeAlias, TypeVar

FieldName: TypeAlias = str
FieldParamSpec: TypeAlias = dict[str]
Icon: TypeAlias = str
PTY_Icon: TypeAlias = str
TTY_Icon: TypeAlias = str

FormatStr: TypeAlias = str

Kwargs: TypeAlias = dict

Bar_T = TypeVar('Bar')


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
        name: FieldName = None,
        func: Callable[..., str] = None,
        icon: str = '',
        fmt: str = None,
        interval: float = 1.0,
        align_to_seconds: bool = False,
        overrides_refresh: bool = False,
        threaded: bool = False,
        always_show_icon: bool = False,
        constant_output: str = None,
        run_once: bool = False,
        bar: Bar_T = None,
        args = None,
        kwargs = None,
        setup: Callable[..., Kwargs] = None,

        # Set this to use different icons for different output streams:
        icons: Sequence[PTY_Icon, TTY_Icon] = None,
    ) -> None:

        if constant_output is None:
            #NOTE: This will change when dynamic icons and fmts are implemented.
            if func is None:
                raise IncompatibleArgsError(
                    f"Either a function that returns a string or "
                    f"a constant output string is required."
                )
            if not callable(func):
                raise TypeError(
                    f"Type of 'func' must be callable, not {type(func)}"
                )

        if name is None and callable(func):
            name = func.__name__
        self.name = name

        self._func = func
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

        if setup is not None and not callable(setup):
            raise TypeError(
                f"Parameter 'setup' must be a callable, not {type(setup)}"
            )
        self._setupfunc = setup
        # self._setupvars = {}

        if icons is None:
            icons = (icon, icon)
        self._icons = icons

        if fmt is None and icons is None:
            raise IncompatibleArgsError(
                "An icon is required when fmt is None."
            )
        self.fmt = fmt

        self._bar = bar

        if asyncio.iscoroutinefunction(func) or threaded:
            self._callback = func
        else:
            # Wrap a synchronous function call:
            self._callback = self._asyncify

        self.align_to_seconds = align_to_seconds
        self.always_show_icon = always_show_icon
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
        params: FieldParamSpec = {},
        source: dict = None
    ):
        '''Used to create default Fields with custom parameters.'''
        if source is None:
            source = cls._default_fields

        default: dict = source.get(name)
        if default is None:
            # return default
            raise DefaultFieldNotFoundError(
                f"{name!r} is not the name of a default Field."
            )

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
    def _format_contents(
        text: str,
        icon: str,
        fmt: FormatStr = None,
        always_show_icon: bool = False
    ) -> str:
        '''A helper function that formats field contents.'''
        if fmt is None:
            if always_show_icon or text:
                return icon + text
            else:
                return text
        else:
            return fmt.format(text, icon=icon)

    async def run(self, once: bool = False) -> None:
        '''Asynchronously run a non-threaded field's callback
        and send updates to the bar.'''
        self._check_bar()
        bar = self._bar
        # Do not run fields which have a constant output;
        # only set their bar buffer.
        if self.constant_output is not None:
            bar._buffers[self.name] = self._format_contents(
                self.constant_output,
                self.icon,
                self.fmt,
                self.always_show_icon
            )
            return

        running = self._bar._can_run.is_set

        func = self._callback
        clock = time.monotonic
        using_format_str = (self.fmt is not None)
        last_val = None

        # Use the pre-defined _setupfunc() to gather constant variables
        # for func() which might only be evaluated at runtime:
        if self._setupfunc is not None:
            try:
                if asyncio.iscoroutinefunction(self._setupfunc):
                    setupvars = (
                        await self._setupfunc(*self.args, **self.kwargs)
                    )
                else:
                    setupvars = self._setupfunc(*self.args, **self.kwargs)

            # If _setupfunc raises FailedSetup with a backup value,
            # use it as the field's new constant_output and update the
            # bar buffer:
            except FailedSetup as e:
                backup = e.backup
                contents = self._format_contents(
                    backup,
                    self.icon,
                    self.fmt,
                    self.always_show_icon
                )
                self.constant_output = contents
                bar._buffers[self.name] = str(contents)
                return

            # On success, give new values to kwargs to pass to func().
            self.kwargs.update(setupvars)

        # Run at least once at the start to ensure the bar is not empty:
        result = await func(*self.args, **self.kwargs)
        last_val = result
        contents = self._format_contents(
            result,
            self.icon,
            self.fmt,
            self.always_show_icon
        )
        bar._buffers[self.name] = contents

        if self.run_once or once:
            return

        if self.align_to_seconds:
            # Sleep until the beginning of the next second.
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))
        start_time = clock()

        # The main loop:
        while running():
            result = await func(*self.args, **self.kwargs)
            # Latency from nonzero execution times causes drift, where
            # sleeps become out-of-sync and the bar skips field updates.
            # This is especially noticeable in fields that update
            # routinely, such as the time.

            # To negate drift, when sleeping until the beginning of the
            # next cycle, we must also compensate for latency.
            await asyncio.sleep(
                # Time until next refresh:
                self.interval - (
                    # Get the current latency, which can vary:
                    clock() % self.interval
                    # (clock() - start_time) % self.interval  # Preserve offset
                )
            )

            # if DEBUG:
                # logger.debug(f"{self.name}: New refresh cycle at {clock() - start_time}")

            if result == last_val:
                continue
            last_val = result

            if using_format_str:
                contents = self.fmt.format(result, icon=self.icon)
            else:
                if self.always_show_icon or result:
                    contents = self.icon + result
                else:
                    contents = result

            bar._buffers[self.name] = contents

            # Send new field contents to the bar's override queue and print a
            # new line between refresh cycles.
            if self.overrides_refresh:
                try:
                    bar._override_queue.put_nowait((self.name, contents))

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
        bar = self._bar
        # Do not run fields which have a constant output;
        # only set their bar buffer.
        if self.constant_output is not None:
            bar._buffers[self.name] = self._format_contents(
                self.constant_output,
                self.icon,
                self.fmt,
                self.always_show_icon
            )
            return

        running = self._bar._can_run.is_set

        clock = time.monotonic
        step = bar._thread_cooldown
        func = self._callback
        using_format_str = (self.fmt is not None)
        last_val = None

        # If the field's callback is asynchronous,
        # it must be run in a new event loop.
        is_async = asyncio.iscoroutinefunction(func)
        local_loop = asyncio.new_event_loop()


        # Use the pre-defined _setupfunc() to gather constant variables
        # for func() which might only be evaluated at runtime:
        if self._setupfunc is not None:
            try:
                if asyncio.iscoroutinefunction(self._setupfunc):
                    setupvars = local_loop.run_until_complete(
                        self._setupfunc(*self.args, **self.kwargs)
                    )
                else:
                    setupvars = self._setupfunc(*self.args, **self.kwargs)

            # If _setupfunc raises FailedSetup with a backup value,
            # use it as the field's new constant_output and update the
            # bar buffer:
            except FailedSetup as e:
                # self._handle_failed_setup(e)
                backup = e.args[0]
                contents = self._format_contents(
                    backup,
                    self.icon,
                    self.fmt,
                    self.always_show_icon
                )
                self.constant_output = contents
                bar._buffers[self.name] = str(contents)
                return

            # On success, give new values to kwargs to pass to func().
            self.kwargs.update(setupvars)

        # Run at least once at the start to ensure the bar is not empty:
        if is_async:
            result = local_loop.run_until_complete(
                func(*self.args, **self.kwargs)
            )
        else:
            result = func(*self.args, **self.kwargs)
        last_val = result
        contents = self._format_contents(
            result,
            self.icon,
            self.fmt,
            self.always_show_icon
        )
        bar._buffers[self.name] = contents

        if self.run_once or once:
            # logger.debug(self.name + str(local_loop.__dict__))
            local_loop.stop()
            local_loop.close()
            self._bar._threads.remove(self._thread)
            return

        count = 0
        needed = round(self.interval / step)
        # if DEBUG:
            # logger.debug(f"{self.name}: {needed = }")

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
                # logger.debug(f"{self.name}: {count = }")
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
                # logger.debug(f"{self.name}: New refresh cycle at {clock() - start_time}")

            count = 0

            if is_async:
                result = local_loop.run_until_complete(
                    func(*self.args, **self.kwargs)
                )
            else:
                result = func(*self.args, **self.kwargs)

            if result == last_val:
                continue
            last_val = result

            if using_format_str:
                contents = self.fmt.format(result, icon=self.icon)
            else:
                if self.always_show_icon or result:
                    contents = self.icon + result
                else:
                    contents = result

            bar._buffers[self.name] = contents

            # Send new field contents to the bar's override queue and print a
            # new line between refresh cycles.
            if self.overrides_refresh:
                # if DEBUG:
                    # logger.debug(f"{self.name}: Sending update @ {clock() - start_time}")
                try:
                    bar._override_queue._loop.call_soon_threadsafe(
                        bar._override_queue.put_nowait, (self.name, contents)
                    )

                except asyncio.QueueFull:
                    # if DEBUG:
                        # logger.debug(f"{self.name}: Failed update: QueueFull")
                    # Since the bar buffer was just updated, do nothing if the
                    # queue is full. The update may still show while the queue
                    # handles the current override.
                    # If not, the line will update at the next refresh cycle.
                    pass

        local_loop.stop()
        local_loop.close()

    def _check_bar(self, no_error: bool = False) -> NoReturn | bool:
        '''Raises MissingBarError if self._bar is None.'''
        if self._bar is None:
            if no_error:
                return False
            raise MissingBarError(
                "Fields cannot run until they belong to a Bar."
            )
        return True

    async def send_to_thread(self, run_once: bool = True) -> None:
        '''Make and start a thread in which to run the field's callback.'''
        self._thread = threading.Thread(
            target=self.run_threaded,
            name=self.name,
            args=(run_once,)
        )
        if self._bar is not None:
            self._bar._threads.add(self._thread)
        self._thread.start()

