#TODO: Implement dynamic icons!
#TODO: Finish Mocp line!


__all__ = (
    'Field',
)


import asyncio
import asyncio.subprocess as aiosp
import os.path
import shlex
import threading
import time
from os import PathLike

from . import field_funcs
from . import _setups
from . import utils
from .constants import DEBUG
from .errors import *
from .formatting import FormatterFieldSig
from .namespaces import FieldSpec
from ._types import (
    Args,
    ASCII_Icon,
    Bar,
    Contents,
    Field,
    FieldFunc,
    FieldFuncSetup,
    FieldName,
    FormatStr,
    Icon,
    Kwargs,
    Unicode_Icon,
)

from collections.abc import Callable, Sequence
from typing import (
    NamedTuple,
    Never,
    Required,
    Self,
    TypeAlias,
    TypeVar
)


class Field:
    '''
    Continuously generate and format one bit of information in a
    :class:`Bar`.

    Pre-defined default Fields can be looked up by name using
    :meth:`Field.from_default`.

    :param name: A unique identifier for the new Field,
        defaults to `func`.:attr:`__name__`
    :type name: :class:`str`

    :param func: The Python function to run at every `interval` if no
        `constant_output` is set
    :type func: :class:`FieldFunc`

    :param icon: The Field icon, positioned in front of Field contents
        or in place of ``{icon}`` in `template`, if provided,
        defaults to ``''``
    :type icon: :class:`str`

    :param template: A curly-brace format string. This parameter is
        **required** if `icon` is ``None``.

        Valid placeholders:
            - ``{icon}`` references `icon`
            - ``{}`` references Field contents

        Example:
            | When the Field's current contents are ``'69F'`` and its\
            icon is ``'TEMP'``,
            | ``template='[{icon}]: {}'`` shows as ``'[TEMP]: 69F'``

    :type template: :class:`FormatStr`

    :param interval: How often in seconds per update Field contents are
        updated, defaults to ``1.0``
    :type interval: :class:`float`

    :param clock_align: Update contents at the start of each second,
        defaults to ``False``
    :type clock_align: :class:`bool`

    :param timely: Run `func` as soon as possible after every refresh,
        defaults to ``False``
    :type timely: :class:`bool`

    :param overrides_refresh: Ensure updates to this Field re-print the
        Bar between refreshes, defaults to ``False``
    :type overrides_refresh: :class:`bool`

    :param threaded: Run this Field in a separate thread,
        defaults to ``False``
    :type threaded: :class:`bool`

    :param always_show_icon: Show icons even when contents are empty,
        defaults to ``False``
    :type always_show_icon: :class:`bool`

    :param run_once: Permanently set contents by running `func` only once,
        defaults to ``False``
    :type run_once: :class:`bool`

    :param constant_output: Permanently set contents instead of running
        a function
    :type constant_output: :class:`str`, optional

    :param bar: Attach the :class:`Field` to this :class:`Bar`
    :type bar: :class:`Bar`, optional

    :param args: Positional args passed to `func`
    :type args: :class:`Args`, optional

    :param kwargs: Keyword args passed to `func`
    :type kwargs: :class:`Kwargs`, optional

    :param setup: A special callback that updates `kwargs` with static
        data that `func` would otherwise have to evaluate every time it
        runs
    :type setup: :class:`FieldFuncSetup`, optional

    :param command: A shell command to run
    :type command: :class:`str`

    :param script: The path to a shell script to run
    :type script: :class:`PathLike`

    :param allow_multiline: Don't join the output of a command or script
        if it spans multiple lines, defaults to ``False``
    :type allow_multiline: :class:`bool`

    :raises: :exc:`errors.IncompatibleArgsError` when
        neither `func` nor `constant_output` are given
    :raises: :exc:`errors.IncompatibleArgsError` when
        neither `icon` nor `template` are given
    :raises: :exc:`TypeError` when `func` is not callable
    :raises: :exc:`TypeError` when `setup`, if given, is not callable
    '''
    _default_fields = {

        'hostname': {
            'name': 'hostname',
            'func': field_funcs.get_hostname,
            'run_once': True
        },

        'host': {
            'name': 'host',
            'func': field_funcs.get_host,
            'kwargs': {
                'fmt': "{nodename}",
            },
            'run_once': True
        },

        'uptime': {
            'name': 'uptime',
            'func': field_funcs.get_uptime,
            'kwargs': {
                'fmt': '{days}d:{hours}h:{mins}m',
                'dynamic': True,
                'sep': ':',
            },
            'setup': _setups.setup_uptime,
            'timely': True,
            'clock_align': True,
            'icon': 'Up ',
        },

        'cpu_usage': {
            'name': 'cpu_usage',
            'func': field_funcs.get_cpu_usage,
            'kwargs': {
                'fmt': "{:02.0f}%",
            },
            'interval': 5,
            'threaded': True,
            'icon': 'CPU ',
        },

        'cpu_temp': {
            'name': 'cpu_temp',
            'func': field_funcs.get_cpu_temp,
            'kwargs': {
                'fmt': "{temp:02.0f}{scale}",
            },
            'interval': 5,
            'threaded': True,
        },

        'mem_usage': {
            'name': 'mem_usage',
            'func': field_funcs.get_mem_usage,
            'kwargs': {
                'fmt': "{used:.1f}{unit}",
                'unit': 'G',
            },
            'interval': 5,
            'icon': 'Mem ',
        },

        'disk_usage': {
            'name': 'disk_usage',
            'func': field_funcs.get_disk_usage,
            'kwargs': {
                'fmt': "{free:.1f}{unit}",
                'path': '/',
                'unit': 'G',
            },
            'interval': 5,
            'icon': '/:',
        },

        'battery': {
            'name': 'battery',
            'kwargs': {
                'fmt': "{pct:02.0f}{state}",
            },
            'func': field_funcs.get_battery_info,
            'threaded': True,
            'icon': 'Bat ',
        },

        'net_stats': {
            'name': 'net_stats',
            'func': field_funcs.get_net_stats,
            'kwargs': {
                # 'device': None,
                'fmt': "{name}",
                'nm': True,
                'nm_filt': None,
                'default': "",
            },
            'interval': 5,
        },

        'datetime': {
            'name': 'datetime',
            'func': field_funcs.get_datetime,
            'kwargs': {
                'fmt': "%Y-%m-%d %H:%M:%S",
            },
            'clock_align': True,
            'timely': True,
        }

    }

    def __init__(
        self,
        *,
        name: FieldName = None,
        func: FieldFunc = None,
        icon: Icon | Sequence[ASCII_Icon, Unicode_Icon] = '',
        template: FormatStr = None,
        interval: float = 1.0,
        clock_align: bool = False,
        timely: bool = False,
        overrides_refresh: bool = False,
        threaded: bool = False,
        always_show_icon: bool = False,
        run_once: bool = False,
        constant_output: str = None,
        bar: Bar = None,
        args: Args = None,
        kwargs: Kwargs = None,
        setup: FieldFuncSetup = None,
        command: str = None,
        script: PathLike = None,
        allow_multiline: bool = False,
    ) -> None:

        if script is not None:
            script = os.path.abspath(script)
            command = self._fetch_script(script)
        if command is not None:
            func = self._run_command
            if script is not None:
                name = os.path.basename(script)
            else:
                name = command.split(None, 1)[0]

        if constant_output is None:
            #NOTE: This will change when dynamic icons and templates are
            # implemented.
            if func is None:
                raise IncompatibleArgsError(
                    f"Either a function `func` that returns a string or "
                    f"a `constant_output` string is required."
                )
            if not callable(func):
                raise TypeError(
                    f"Type of `func` must be callable, not {type(func)}"
                )

        if name is None and callable(func):
            name = func.__name__
        self.name = name

        self._func = func
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

        if setup is not None and not callable(setup):
            # `setup` was given but is of the wrong type.
            raise TypeError(
                f"Type of 'setup' must be callable, not {type(setup)}"
            )
        self._setupfunc = setup

        if isinstance(icon, str):
            icon = (icon, icon)
        elif not isinstance(icon, Sequence):
            raise TypeError("`icon` must be a sequence")
        self._icons = icon

        if template is None and icon is None:
            raise IncompatibleArgsError(
                "An icon is required when `template` is None."
            )
        self.template = template

        self.is_async = asyncio.iscoroutinefunction(func)

        if self.is_async or threaded or timely:
            self._callback = func
        else:
            # Wrap synchronous functions if they aren't special:
            self._callback = self._asyncify

        self.allow_multiline = allow_multiline
        self.always_show_icon = always_show_icon
        self._bar = bar
        self._buffer = None
        self.clock_align = clock_align
        self.command = command
        self.constant_output = constant_output
        self.interval = interval
        self.overrides_refresh = overrides_refresh
        self.run_once = run_once
        self.script = script
        self.threaded = threaded
        self.timely = timely

        self._do_setup()

    def __repr__(self) -> str:
        cls = type(self).__name__
        name = self.name
        # attrs = utils.join_options(...)
        return f"{cls}({name=})"

    def __eq__(self, other: Self) -> bool:
        if not all(
            getattr(self, attr) == getattr(other, attr)
            for attr in (
                'clock_align',
                'always_show_icon',
                'constant_output',
                'template',
                '_icons',
            )
        ):
            return False

        if self._func is None:
            if self.constant_output == other.constant_output:
                return True
        elif (
            self._func is other._func
            and self.args == other.args
            and self.kwargs == other.kwargs
        ):
            return True
        return False

    @classmethod
    def from_default(
        cls: Self,
        name: str,
        *,
        overrides: FieldSpec = {},
        source: dict[FieldName, FieldSpec] = None,
        fmt_sig: FormatStr = None
    ) -> Self:
        '''Quickly get a default Field and customize its parameters.

        :param name: Name of the default :class:`Field` to access or customize
        :type name: :class:`str`

        :param overrides: Custom parameters that override those of the default Field
        :type overrides: :class:`namespaces.FieldSpec`, optional

        :param source: The :class:`dict` in which to look up default fields,
            defaults to :attr:`Field._default_fields`
        :type source: dict[:class:`FieldName`, :class:`namespaces.FieldSpec`]

        :returns: A new :class:`Field`
        :rtype: :class:`Field`
        :raises: :exc:`errors.DefaultFieldNotFoundError` when `source` does not contain `name`
        '''
        if source is None:
            source = cls._default_fields

        try:
            default: FieldSpec = source[name].copy()
        except KeyError:
            raise DefaultFieldNotFoundError(
                f"{name!r} is not the name of a default Field."
            ) from None

        if overrides.get('interval', None) != default.get('interval', None):
            overrides['timely'] = False
        if 'kwargs' in overrides and 'kwargs' in default:
            overrides['kwargs'] = default['kwargs'] | overrides['kwargs']

        spec = default | overrides
        field = cls(**spec)
        field._fmt_sig = fmt_sig
        return field

    @classmethod
    def from_format_string(cls, fmt: FormatStr) -> Self:
        '''
        Get a :class:`Field` from a curly-brace field in a format string.

        :param template: The format string to convert
        :type template: :class`FormatStr`
        '''
        sig = FormatterFieldSig.from_str(fmt)
        field = cls.from_default(sig.name)
        field._fmtsig = fmt
        return field

    @property
    def icon(self) -> str:
        '''The field icon as determined by the output stream of its bar.
        It defaults to the TTY icon (:attr:`self._icon[1]`) if no bar is set.
        '''
        if self._bar is None:
            # Default to using the ASCII icon:
            return self._icons[0]
        unicode = self._bar._unicode or not self._bar._stream.isatty()
        return self._icons[unicode]

    async def _asyncify(self, *args, **kwargs) -> Contents:
        '''Wrap a synchronous function in a coroutine for simplicity.'''
        return self._func(*args, **kwargs)

    def _fetch_script(self, script: PathLike) -> str:
        with open(script, 'r') as f:
            script = f.read()
        return script

    async def _run_command(self, *args, **kwargs) -> Contents:
        cmd = shlex.join(shlex.split(self.command))
        proc = await aiosp.create_subprocess_shell(
            cmd,
            stdout=aiosp.PIPE,
            stderr=aiosp.PIPE
        )
        out, err = await proc.communicate()
        result = out.decode()
        if not self.allow_multiline:
            result = ' '.join(result.splitlines())
        return result

    @staticmethod
    def _format_contents(
        text: str,
        icon: str,
        template: FormatStr = None,
        always_show_icon: bool = False
    ) -> Contents:
        '''A helper function that formats field contents.'''
        if template is None:
            if always_show_icon or text:
                return icon + text
            else:
                return text
        else:
            return template.format(text, icon=icon)

    def _auto_format(self, text: str) -> Contents:
        '''Non-staticmethod _format_contents...'''
        if self.template is None:
            if self.always_show_icon or text:
                return self.icon + text
            else:
                return text
        else:
            return self.template.format(text, icon=self.icon)

    def _do_setup(self, args: Args = None, kwargs: Kwargs = None) -> None:
        '''
        Use the pre-defined _setupfunc() to gather constant variables
        for func() which might only be evaluated at runtime.
        '''
        if self._setupfunc is None:
            return

        if args is None:
            args = self.args
        if kwargs is None:
            kwargs = self.kwargs

        try:
            if asyncio.iscoroutinefunction(self._setupfunc):
                setupvars = asyncio.get_event_loop().run_until_complete(
                    self._setupfunc(*self.args, **self.kwargs)
                )
            else:
                setupvars = self._setupfunc(*self.args, **self.kwargs)

        # If _setupfunc raises FailedSetup with a backup value,
        # use it as the field's new constant_output:
        except FailedSetup as e:
            backup = e.backup
            self.constant_output = self._auto_format(backup)
            return

        # On success, give new values to kwargs to pass to _func().
        # return setupvars
        kwargs['setupvars'] = setupvars

    def sync_run(self, once: bool = None) -> Contents:
        # Do not run fields which have a constant output;
        # only set their bar buffer.
        if self.constant_output is not None:
            return self._auto_format(self.constant_output)
        if self.is_async:
            result = asyncio.get_event_loop().run_until_complete(
                self._func(*self.args, **self.kwargs)
            )
        else:
            result = self._func(*self.args, **self.kwargs)
        contents = self._auto_format(result)
        return contents

    def gen_run(self, once: bool = None) -> Contents:
        # Do not run fields which have a constant output;
        # only set their bar buffer.
        if self.constant_output is not None:
            return self._auto_format(self.constant_output)
        while True:
            if self.is_async:
                result = self._func(*self.args, **self.kwargs)
            else:
                result = asyncio.get_event_loop().run_until_complete(
                    self._func(*self.args, **self.kwargs)
                )
            contents = self._auto_format(result)
            yield contents

    async def run(self, bar: Bar, once: bool) -> None:
        '''
        Run an async :class:`Field` callback and send its output to a
        status bar.

        :param bar: Send results to this status bar
        :type bar: :class:`Bar`

        :param once: Run the Field function once
        :type once: :class:`bool`
        '''
        self._check_bar(bar)
        # Do not run fields which have a constant output;
        # only set their bar buffer.
        if self.constant_output is not None:
            contents = self._format_contents(
                self.constant_output,
                self.icon,
                self.template,
                self.always_show_icon
            )
            bar._buffers[self.name] = contents
            return

        func = self._callback
        running = bar._can_run.is_set
        clock = time.monotonic
        using_format_str = (self.template is not None)
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
            # bar's buffer:
            except FailedSetup as e:
                backup = e.backup
                contents = self._format_contents(
                    backup,
                    self.icon,
                    self.template,
                    self.always_show_icon
                )
                self.constant_output = contents
                bar._buffers[self.name] = str(contents)
                return

            # On success, give new values to kwargs to pass to func().
            self.kwargs['setupvars'] = setupvars

        # Run at least once at the start to ensure bars have contents:
        result = await func(*self.args, **self.kwargs)
        last_val = result
        contents = self._auto_format(result)
        bar._buffers[self.name] = contents

        if self.run_once or once:
            return

        if self.clock_align:
            # Sleep until the beginning of the next second.
            clock = time.time
            await asyncio.sleep(1 - (clock() % 1))
        start_time = clock()

        # The main loop:
        while running():

            if bar.count:
                # Stop before running a cycle that could last too long:
                if (
                    bar._print_countdown == 0
                    or self.interval > (
                        bar._print_countdown * bar.refresh_rate 
                    )
                ):
                    bar._coros.pop(self.name, None)
                    return

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

            if result == last_val:
                continue
            last_val = result

            if using_format_str:
                contents = self.template.format(result, icon=self.icon)
            else:
                if self.always_show_icon or result:
                    contents = self.icon + result
                else:
                    contents = result

            bar._buffers[self.name] = contents

            # Send new field contents to the bar's override queue and
            # print a new line between refresh cycles.
            if self.overrides_refresh:
                try:
                    bar._override_queue.put_nowait((self.name, contents))

                except asyncio.QueueFull:
                    # Since the bar buffer was just updated, do nothing
                    # if the queue is full. The update may still show
                    # while the queue handles the current override. If
                    # not, the line will update at the next refresh cycle.
                    pass

    def run_threaded(self, bar: Bar, once: bool) -> None:
        '''
        Run a blocking :class:`Field` func and send its output to a
        status bar.

        :param bar: Send results to this status bar
        :type bar: :class:`Bar`

        :param once: Run the Field function once
        :type once: :class:`bool`
        '''
        self._check_bar(bar)

        # Do not run fields which have a constant output;
        # only set their bar buffer.
        if self.constant_output is not None:
            contents = self._format_contents(
                self.constant_output,
                self.icon,
                self.template,
                self.always_show_icon
            )
            bar._buffers[self.name] = contents
            return

        func = self._callback
        running = bar._can_run.is_set
        clock = time.monotonic
        using_format_str = (self.template is not None)
        last_val = None

        # If the field's callback is asynchronous,
        # it must be run in a new event loop.
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
                backup = e.args[0]
                contents = self._format_contents(
                    backup,
                    self.icon,
                    self.template,
                    self.always_show_icon
                )
                self.constant_output = contents
                bar._buffers[self.name] = str(contents)
                return

            # On success, give new values to kwargs to pass to func().
            self.kwargs['setupvars'] = setupvars

        # Run at least once at the start to ensure bar is not empty:
        if self.is_async:
            result = local_loop.run_until_complete(
                func(*self.args, **self.kwargs)
            )
        else:
            result = func(*self.args, **self.kwargs)
        last_val = result
        contents = self._format_contents(
            result,
            self.icon,
            self.template,
            self.always_show_icon
        )
        bar._buffers[self.name] = contents

        if self.run_once or once:
            local_loop.stop()
            local_loop.close()
            bar._threads.pop(self.name)  #NOTE: Eventually, id(self)
            return

        step = bar._thread_cooldown
        needed = round(self.interval / step)
        count = 0
        first_cycle = True

        if self.clock_align:
            # Sleep until the beginning of the next second.
            clock = time.time
            time.sleep(1 - (clock() % 1))

        start_time = clock()
        while running():

            if bar.count:
                # Stop before running a cycle that could last too long:
                if (
                    bar._print_countdown == 0
                    or self.interval > (
                        bar._print_countdown * bar.refresh_rate 
                    )
                ):
                    local_loop.stop()
                    local_loop.close()
                    bar._coros.pop(self.name, None)
                    return


            # Sleep until the next refresh cycle in little steps to
            # check if the bar has stopped.
            if count < needed and not first_cycle:
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
                        # (clock() - start_time) % step To preserve offset
                    )
                )
                continue

            count = 0
            if first_cycle:
                first_cycle = False

            if self.is_async:
                result = local_loop.run_until_complete(
                    func(*self.args, **self.kwargs)
                )
            else:
                result = func(*self.args, **self.kwargs)

            if result == last_val:
                continue
            last_val = result

            if using_format_str:
                contents = self.template.format(result, icon=self.icon)
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
                    bar._override_queue._loop.call_soon_threadsafe(
                        bar._override_queue.put_nowait, (self.name, contents)
                    )
                except asyncio.QueueFull:
                    # Since the bar buffer was just updated, do nothing
                    # if the queue is full. The update may still show
                    # while the queue handles the current override. If
                    # not, the line will update at the next refresh cycle.
                    pass

        local_loop.stop()
        local_loop.close()

    @staticmethod
    def _check_bar(
        bar: Bar,
        raise_on_fail: bool = True
    ) -> Never | bool:
        '''
        Check if a bar has the right attributes to use a run() function.
        If these attributes are missing,
            return ``False`` if `raise_on_fail` is ``True``,
            or raise :exc:`InvalidBarError` if `raise_on_fail` is ``False``.

        :param bar: The status bar object to test
        :type bar: :class:`Bar`

        :param raise_on_fail: Raise an exception if `bar` fails the check,
            defaults to ``False``
        :type raise_on_fail: :class:`bool`

        :raises: :exc:`InvalidBarError` if the stream fails the test
            and lacks required attributes
        '''
        required_attrs = (
            '_buffers',
            '_can_run',
            '_override_queue',
            '_stream',
        )
        if not all(hasattr(bar, a) for a in required_attrs):
            if not raise_on_fail:
                return False
            raise InvalidBarError(
                f"Status bar {bar!r} is missing certain attributes"
                f" required to run `Field.run()`"
                f" and `Field.run_threaded()`."
            ) from None
        return True

    def make_thread(self, bar: Bar, run_once: bool) -> None:
        '''
        Return a thread that runs the :class:`Field`'s callback.

        :param bar: Send results to this status bar
        :type bar: :class:`Bar`

        :param once: Run the Field function once
        :type once: :class:`bool`
        '''
        thread = threading.Thread(
            target=self.run_threaded,
            args=(bar, run_once),
            name=self.name  # #NOTE Eventually, id(self)
        )
        return thread


type FieldPrecursor = FieldName | Field | FormatterFieldSig
from . import _types
FieldPrecursor = FieldPrecursor

