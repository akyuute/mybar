#TODO: Add Bar.fields property
#TODO: Field.is_running: attr or property that calls threading.Event().is_set()
#TODO: Implement align_to_seconds!
#TODO: Implement icon tuples!
#TODO: DescriptiveTypeName = str for type hints!
#TODO: Command line args!
#TODO: Finish Mocp line!
#TODO: Implement dynamic icons!


import asyncio
import inspect
import json
import os
import sys
import threading

from argparse import ArgumentParser
from string import Formatter
from time import sleep, monotonic #, time
from typing import Callable, IO, Iterable

from mybar import field_funcs
from mybar import setups
from mybar.errors import *
from mybar.utils import join_options, str_to_bool, clean_comment_keys

__all__ = (
    'Field',
    'Bar',
    'Config',
    'run',
)

CONFIG_FILE = '~/.mybar.json'
CSI = '\033['  # Unix terminal escape code (control sequence introducer)
CLEAR_LINE = '\x1b[2K'  # VT100 escape code to clear line
HIDE_CURSOR = '?25l'
UNHIDE_CURSOR = '?25h'
COUNT = [0]


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
            'icon': 'Up ',
            'kwargs': {
                'fmt': '{days}d:{hours}h:{mins}m',
                'sep': ':'
            },
        },

        'cpu_usage': {
            'name': 'cpu_usage',
            'func': field_funcs.get_cpu_usage,
            'interval': 2,
            'threaded': True
        },

        'cpu_temp': {
            'name': 'cpu_temp',
            'func': field_funcs.get_cpu_temp,
            'interval': 2,
            'threaded': True,
            'icon': 'CPU ',
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
            'interval': 1,
            'overrides_refresh': True,
            'icon': 'Bat '
        },

        'net_stats': {
            'name': 'net_stats',
            'func': field_funcs.get_net_stats,
            'interval': 4
        },

        'datetime': {
            'name': 'datetime',
            'func': field_funcs.get_datetime,
        },

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
        args = None,
        kwargs = None,
        setup: Callable = None,


        # icons: Iterable[str] = None,
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
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs

        if setup is not None:
            if not callable(setup):
                raise TypeError(
                    f"Parameter 'setup' must be a callable, not {type(setup)}")
        self._setupfunc = setup
        # self._setupvars = {}

        if fmt is None:
            if all(s is None for s in (gui_icon, term_icon, icon)):
                raise IncompatibleParams(
                    "An icon is required when fmt is None.")
        self.fmt = fmt

        self._bar = bar

        self.term_icon = term_icon
        self.gui_icon = gui_icon
        self.default_icon = icon
        self.icon = self.get_icon(icon)

        if inspect.iscoroutinefunction(func) or threaded:
            self._callback = func
        else:
            # Wrap a synchronous function call
            self._callback = self._asyncify

        # self.align_to_seconds = align_to_seconds
        self._buffer = None
        self.constant_output = constant_output
        self.interval = interval
        self.overrides_refresh = overrides_refresh
        self.run_once = run_once

        self.threaded = threaded
        self._thread = None
        # self.is_running = False

    def __repr__(self):
        cls = type(self).__name__
        name = self.name
        # fields = self.fields
        # attrs = join_options(...)
        return f"{cls}({name=})"

    @classmethod
    def from_default(cls,
        name: str,
        params: dict = {},
        source: dict = None
    ):
        '''Used to create default Fields with custom parameters.'''
        if source is None:
            source = cls._default_fields

        default: dict = source.get(name)
        if default is None:
            raise DefaultNotFound(
                f"{name!r} is not the name of a default Field.")

        spec = {**default}
        spec.update(params)
        return cls(**spec)

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

        # Defining local variables imperceptibly improves performance.
        # (Fewer LOAD_ATTRs, more LOAD_FASTs.)
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
        args = self.args
        kwargs = self.kwargs

        icon = self.get_icon(self.default_icon)
        fmt = self.fmt
        using_format_str = (fmt is not None)
        last_val = None

        # Use self.setup() to gather static variables which need to
        # be evaluated at runtime and passed to self._func.
        if self._setupfunc is not None:
            try:
                kwargs.update(
                    await self.setup(args, kwargs)
                )

            except FailedSetup as e:
                backup = e.args[0]
                contents = await self.add_icon(
                    backup,
                    icon,
                    using_format_str,
                    fmt
                )
                field_buffers[field_name] = self.constant_output = contents
                return

        # The main loop:
        start_time = monotonic()
        while running.is_set():
            res = await func(*args, **kwargs)

            if res == last_val:
                await asyncio.sleep(interval)
                continue

            last_val = res

            if using_format_str:
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

                # Running the bar a second time raises RuntimeError saying its
                # event loop is closed.
                # It's usually not, so we ignore that.
                except RuntimeError:
                    # print("\nGot RuntimeError")
                    # assert self._bar._loop.is_running()
                    pass

            if run_once:
                return

            # "Drift will cause the output of a field with values that
            # change routinely (such as the time) to update out of sync
            # with the changes to its value.
            # Sleep until the next cycle, and compensate for drift
            # caused by nonzero execution times:
            await asyncio.sleep(
                # Time until next refresh:
                interval - (
                    # Get the current drift, which can vary:
                    (monotonic() - start_time)
                    % interval
                )
            )

    def run_threaded(self):
        '''Run a blocking function in a thread
        and send its updates to the bar.'''
        self._check_bar()

        # Defining local variables imperceptibly improves performance.
        # (Fewer LOAD_ATTRs, more LOAD_FASTs.)
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
        args = self.args
        kwargs = self.kwargs

        icon = self.get_icon(self.default_icon)
        fmt = self.fmt
        using_format_str = (fmt is not None)
        last_val = None

        cooldown = bar._thread_cooldown

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
                contents = loop.run_until_complete(
                    self.add_icon(
                        backup,
                        icon,
                        using_format_str,
                        fmt
                    )
                )
                field_buffers[field_name] = self.constant_output = contents
                return

        # The main loop:
        count = 0
        start_time = monotonic()
        while running.is_set():
            # Rather than block for the whole interval,
            # use tiny cooldowns to check if the bar is still running.
            # A shorter cooldown means more chances to check if the bar stops.
            # Thus, threads usually cancel `cooldown` seconds after self._func
            # returns when the bar stops rather than after `interval` seconds.
            if count != round(interval / cooldown):
                count += 1
                # "Drift will cause the output of a field with values that
                # change routinely (such as the time) to update out of sync
                # with the changes to its value.
                # Sleep until the next cycle, and compensate for drift
                # caused by nonzero execution times:
                sleep(
                    cooldown - (
                        # Get the current drift, which can vary:
                        (monotonic() - start_time)
                        % cooldown
                    )
                )
                continue

            if is_async:
                res = loop.run_until_complete(func(*args, **kwargs))
            else:
                res = func(*args, **kwargs)

            if res == last_val:
                count = 0
                continue

            last_val = res

            if using_format_str:
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
                return

            count = 0

        loop.stop()
        loop.close()

    def _check_bar(self):
        '''Raises MissingBar if self._bar is None.'''
        if self._bar is None:
            raise MissingBar("Fields cannot run until they belong to a Bar.")

    async def setup(self, args, kwargs):
        '''Initialize static variables used by self._func which would
        otherwise be evaluated at each iteration.'''
        if inspect.iscoroutinefunction(self._setupfunc):
            return await self._setupfunc(*args, **kwargs)
        return self._setupfunc(*args, **kwargs)

    async def add_icon(self,
        text: str,
        icon: str,
        using_format_str: bool,
        fmt: str = None
    ) -> str:
        '''A helper function to add an icon to the output of a field.'''
        if using_format_str:
            contents = fmt.format(text, icon=icon)
        else:
            contents = icon + text
        return contents

    async def send_to_thread(self):
        '''Make and start a thread in which to run the field's callback.'''
        self._thread = threading.Thread(
            target=self.run_threaded,
            name=self.name
        )
        self._thread.start()


class Bar:

    _default_params = {
        'separator': '|',
        'refresh_rate': 1.0,
        'join_empty_fields': False,

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
        *,
        fields: Iterable[Field|str] = None,
        fmt: str = None,
        separator: str = '|',
        refresh_rate: float = 1.0,
        stream: IO = sys.stdout,
        join_empty_fields: bool = False,
        gui_sep: str = None,
        term_sep: str = None,
        override_cooldown: float = 1/60,
        thread_cooldown: float = 1/8
    ):
        # Ensure the output stream has the required methods.
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

            if all(s is None for s in (gui_sep, term_sep, separator)):
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
        self.term_sep = term_sep
        self.gui_sep = gui_sep
        self.separator = self.get_separator(separator)
        self.default_sep = separator

        # Whether empty fields are joined when fmt is None:
        # (True shows two separators together for every blank field.)
        self.join_empty_fields = join_empty_fields

        # Set how often in seconds the bar is normally printed.
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

    def __repr__(self):
        names = self._field_order
        fields = join_options(names, final_sep='', quote=True, limit=3)
        cls = type(self).__name__
        return f"{cls}(fields=[{fields}])"

    @property
    def fields(self):
        return tuple(self._fields.values())

    @property
    def in_a_tty(self) -> bool:
        if self._stream is None:
            return False
        return self._stream.isatty()

    @property
    def clearline_char(self) -> str:
        '''Terminal streams print this between refreshes, preventing
        longer lines from leaving behind characers.'''
        if self._stream is None:
            return None
        clearline = CLEAR_LINE if self._stream.isatty() else ''
        return clearline

    @classmethod
    def from_dict(cls, dct: dict):
        '''Accept a mapping of Bar parameters.'''
        #NOTE: This can raise AttributeError if it encounters a nested list!
        data = clean_comment_keys(dct, '//')
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
                        raise UndefinedField(f"\n"
                            f"In 'field_order':\n"
                            f"  Expected the name of a default field or a "
                            f"custom field defined in 'field_definitions', "
                            f"but got {name!r} instead."
                        )

                case {'custom': True}:
                    # The field is custom, so it is only defined in
                    # 'field_definitions' and will not be found as a default.
                    custom_name = params.pop('name', name)
                    field = Field(**params, name=custom_name)

                case {}:
                    # The field is a default overridden by the user.
                    field = Field.from_default(name, params)
                    if field is None:
                        raise DefaultNotFound(f"\n"
                            f"In 'field_definitions':\n"
                            f"  Expected a default Field name but got {name!r} "
                            f"instead.\n"
                            f"  For JSON, remember to specify `\"custom\": true` "
                            f"in custom field definitions."
                        )

                case _:
                    print("Got a weird field.")
                    print(f"{name = }, {params = }")

            fields.append(field)

        return cls(fields=fields, **bar_params)

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
            err = f"Invalid bar format string: {fmt!r}"
            raise BrokenFormatString(err) from None
        if '' in field_names:
            raise BrokenFormatString(
                "The bar format string cannot contain "
                "positional fields ('{}').")
        return field_names

    def convert_fields(self, fields: Iterable[str] = None) -> dict[str, Field]:
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

    def get_separator(self, default=None) -> str:
        if self._stream is None:
            return default
        sep = self.term_sep if self._stream.isatty() else self.gui_sep
        if sep is None:
            sep = default
        return sep

    def run(self, stream: IO = None):
        '''Run the bar.
        Block until an exception is raised and exit smoothly.'''
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
            # Do not run fields which have a constant output,
            # only set their bar buffer.
            if (output := field.constant_output) is not None:
                if field.fmt is None:
                    self._buffers[field.name] = output
                else:
                    self._buffers[field.name] = field.fmt.format(output)

            elif field.threaded:
                await field.send_to_thread()

            else:
                field_coros.append(field.run())

        await asyncio.gather(
            *field_coros,
            self._handle_overrides(),
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
        running = self._can_run
        refresh = self.refresh_rate

        if self.in_a_tty:
            beginning = clearline + end
            stream.write(CSI + HIDE_CURSOR)
        else:
            beginning = clearline

        # Flushing the buffer before writing to it fixes poor i3bar alignment.
        stream.flush()

        start_time = monotonic()
        while running.is_set():
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
                    # Get the current drift, which can vary:
                    (monotonic() - start_time)
                    % refresh
                )
            )

    async def _handle_overrides(self, end: str = '\r'):
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
        running = self._can_run
        refresh = self.refresh_rate
        override_queue = self._override_queue
        cooldown = self._override_cooldown

        if self.in_a_tty:
            beginning = clearline + end
        else:
            beginning = clearline

        while running.is_set():
            try:
                # Wait until a field that overrides the refresh rate
                # (with overrides_refresh) sends new data to be printed:
                field, contents = await override_queue.get()
            except RuntimeError:
                # asyncio.queues raises an error when getter.cancel()
                # is called and the event loop is closed.
                # It serves no purpose when the bar has just stopped.
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
            await asyncio.sleep(cooldown)

class Config:
    def __init__(self, file: os.PathLike = None): # Path = None):
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

    def __repr__(self):
        cls = self.__class__.__name__
        file = self.file
        return f"{cls}({file=})"

    def make_bar(self) -> Bar:
        return Bar.from_dict(self.data)

    def read_file(self, file: os.PathLike = None) -> tuple[dict, str]:
        if file is None:
            file = self.file
        with open(self.file, 'r') as f:
            data = json.load(f)
            text = f.read()
        return data, text

    def write_file(self, file: os.PathLike = None, obj: dict = None):
        if file is None:
            file = self.file

        # return json.dumps(obj)
        obj = Bar._default_params.copy() if obj is None else obj

        dft_bar = obj
        dft_fields = Field._default_fields.copy()

        for name, field in dft_fields.items():
            new = dft_fields[name] = field.copy()
            for param in ('func', 'setup'):
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


def run():

##    fdate = Field(
##        # fmt="<3 {icon}{}!",
##        name='datetime',
##        func=field_funcs.get_datetime,
##        interval=1/8,
##        overrides_refresh=True,
##        term_icon='?'
##    )

##    fcounter = Field(name='counter', func=counter, interval=1, args=[COUNT])

##    ftty = Field(
##        name='isatty',
##        func=(lambda args=None, kwargs=None: str(bar.in_a_tty)),
##        # icon='TTY:',
##        run_once=True,
##        # interval=0
##    )

    # bar = Bar(fields='uptime cpu_usage cpu_temp mem_usage datetime datetime datetime disk_usage battery net_stats datetime'.split())


    # CFG = Config('bar.json')
    CFG = Config(CONFIG_FILE)
    bar = CFG.make_bar()
##    if not os.path.exists(CONFIG_FILE):
##        CFG.write_file()

    bar.run()

