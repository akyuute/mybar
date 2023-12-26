__all__ = (
    # 'get_audio_volume'
    'get_battery_info',
    'get_cpu_temp',
    'get_cpu_usage',
    'get_datetime',
    'get_disk_usage',
    'get_host',
    'get_hostname',
    'get_mem_usage',
    'get_net_stats',
    'get_uptime',
)


import os
import re
import shlex
import time
from asyncio import subprocess as aiosp
from datetime import datetime
from string import Formatter

from .errors import InvalidArgError
from .formatting import ElapsedTime, ConditionalFormatStr, format_uptime
from .utils import join_options
from ._types import (
    POWERS_OF_1024,
    Contents,
    FormatStr,
    MetricSymbol,
    NmConnFilterSpec,
    StorageMeasure,
)

from collections.abc import Callable, Iterable
from typing import Literal, TypeAlias, NamedTuple, Any
from enum import Enum


# Check if all field functions will work:
try:
    os.stat('/proc/stat')
except PermissionError:
    raise CompatibilityWarning(
        "Field functions which require access to procfs will break"
        " on this system."
    )
else:
    import psutil


async def get_audio_volume(
    fmt: FormatStr = "{:02.0f}{state}",
    *args, **kwargs
) -> Contents:
    '''
    Currently awaiting a more practical implementation that supports
    instantaneous updates. A socket would be quite nice for that.
    '''
##    '''Returns system audio volume from ALSA amixer. SIGUSR1 is used to
##    notify the thread of volume changes from button presses.'''
    pat = re.compile(r'.*\[(\d+)%\] \[(\w+)\]')

    cmd = await aiosp.create_subprocess_shell(
        "amixer sget Master",
        stdout=aiosp.PIPE
    )
    out, err = await cmd.communicate()
    state_list = (l.strip() for l in reversed(out.decode().splitlines()))

    pcts, states = zip(*(
        m.groups()
        for l in state_list
        if (m := pat.match(l))
    ))
    avg_pct = sum(int(p) for p in pcts) // len(pcts)
    is_on = any(s == 'on' for s in states)
    # return is_on, avg_pct
    state = "" if is_on else "M"
    return fmt.format(avg_pct, state=state)


##def ctx_get_battery_info(
##    fmt: FormatStr = "{icon}{pct:02.0f}",
##    # fmt: FormatStr = "{icon}{pct:02.0f}{state}",
##    *args, **kwargs
##) -> Context:
##    '''
##    Battery capacity as a percent and whether or not it is charging.
##    '''
##
##    # if not (battery := psutil.sensors_battery()):
##    battery = psutil.sensors_battery()
##    if not battery:
##        return Context()
####        # return (None, None)
##    # state = "CHG" if battery.power_plugged else ''
##
##    # Progressive/dynamic battery icons!
##    icon_bank = "    ".split()
##    # return icon_bank
##
##    def mapper(n):
##        icon = ""
##        for i, test in enumerate((10, 25, 50, 75, 100)):
##            if n <= test:
##                icon = icon_bank[i]
##                return icon + " "
##
##    if battery.power_plugged:
##        icon = ""
##        # state = "CHG"
##    else:
##        icon = mapper(battery.percent)
##        # state = ""
##    # print(repricon)
##
##    info = fmt.format_map({'icon': icon, 'pct': battery.percent, 'state': battery.power_plugged})
####    return battery.power_plugged, info
##    return info


async def get_battery_info(
    fmt: FormatStr = "{pct:02.0f}{state}",
    *args, **kwargs
) -> Contents:
    '''
    Battery capacity as a percent and whether or not it is charging.

    :param fmt: A curly-brace format string with\
        two optional named fields:
            - ``pct``: Current battery percent as a :class:`float`
            - ``state``: Whether or not the battery is charging

        Defaults to ``"{pct:02.0f}{state}"``
    :type fmt: :class:`FormatStr`
    '''
    battery = psutil.sensors_battery()
    if not battery:
        return ""
    state = "CHG" if battery.power_plugged else ''
    info = fmt.format_map({'pct': battery.percent, 'state': state})
    return info


def get_cpu_temp(
    fmt: str = "{temp:02.0f}{scale}",
    in_fahrenheit=False,
    *args, **kwargs
) -> Contents:
    '''
    Current CPU temperature in Celcius or Fahrenheit.

    :param fmt: A curly-brace format string with\
        two optional named fields:
            - ``temp``: Current CPU temperature as a :class:`float`
            - ``scale``: ``'C'`` or ``'F'``, depending on `in_fahrenheit`

        Defaults to ``"{temp:02.0f}{scale}"``
    :type fmt: :class:`FormatStr`

    :param in_fahrenheit: Display the temperature in Fahrenheit instead
        of Celcius, defaults to ``False``
    :type in_fahrenheit: :class:`bool`
    '''
    scales = ('C', 'F')
    scale = scales[in_fahrenheit]

    temps = psutil.sensors_temperatures(in_fahrenheit)
    match temps:
        case {'k10temp': t} | {'coretemp': t}:
            current = t[0].current
        case _:
            current = '??'
    return fmt.format_map({'temp': current, 'scale': scale})


def get_cpu_usage(
    fmt: FormatStr = "{:02.0f}%",
    interval: float = None,
    *args, **kwargs
) -> Contents:
    '''
    System CPU usage in percent over a specified interval.

    :param fmt: A curly-brace format string with one positional field,
        defaults to ``"{:02.0f}%"``
    :type fmt: :class:`FormatStr`

    :param interval: Time to block before returning a result.
        Only set this in a threaded :class:`Field`.
    :type interval: :class:`float`, optional
    '''
    return fmt.format(psutil.cpu_percent(interval))


def get_datetime(
    fmt: str = "%Y-%m-%d %H:%M:%S",
    *args, **kwargs
) -> Contents:
    '''
    Current time as formatted with `fmt`.

    :param fmt: A format string with %-based format codes used by
        ``datetime.strftime()``, defaults to ``"%Y-%m-%d %H:%M:%S"``
    :type fmt: :class:`str`

    .. seealso:: `Datetime strftime() format codes\
    <https://docs.python.org/3/library/datetime.html#strftime-and-strptime\
    -format-codes>`_

    '''
    return datetime.now().strftime(fmt)


async def get_disk_usage(
    fmt: FormatStr = "{free:.1f}{unit}",
    path: os.PathLike = '/',
    unit: MetricSymbol = 'G',
    *args, **kwargs
) -> Contents:
    '''
    Disk usage of a partition at a given path.

    :param fmt: A curly-brace format string with\
        five optional named fields:
            - ``unit``: The same value as `unit`
            - ``total``: Total disk partition size
            - ``used``: Used disk space
            - ``free``: Free disk space (``total - used``)
            - ``percent``: Percent of total disk space that is used\
                (``used / total``, a :class:`float`)

        Defaults to ``"{free:.1f}{unit}"``
    :type fmt: :class:`FormatStr`

    :param path: The path to a directory or file on a disk partition,
        defaults to ``'/'``
    :type path: :class:`os.PathLike`

    :param unit: The unit prefix symbol, defaults to ``'G'``
    :type unit: :class:`MetricSymbol`

    :raises: :exc:`errors.InvalidArgError`
        if `unit` is not a valid :class:`MetricSymbol`
    '''
    if unit not in POWERS_OF_1024:
        raise InvalidArgError(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(POWERS_OF_1024)}."
        )

    disk = psutil.disk_usage(path)
    factor = POWERS_OF_1024[unit]
    converted = {
        meas: (val if meas == 'percent' else val/factor)
        for meas, val in disk._asdict().items()
    }
    converted['unit'] = unit
    usage = fmt.format_map(converted)
    return usage


def get_host(
    fmt: FormatStr = "{nodename}",
    *args, **kwargs
) -> Contents:
    '''
    System host information using :func:`os.uname()`.

    :param fmt: A curly-brace format string with\
        five optional named fields:
            - ``nodename``: System hostname
            - ``sysname``: Operating system kernel name
            - ``release``: Kernel release
            - ``version``: Kernel version
            - ``machine``: Machine architecture

        Defaults to ``"{nodename}"``
    :type fmt: :class:`FormatStr`
    '''
    keys = HostOption.__args__
    host = fmt.format_map(dict(zip(keys, os.uname())))
    return host


def get_hostname(*args, **kwargs) -> Contents:
    '''System hostname.'''
    return os.uname().nodename


def get_mem_usage(
    fmt: FormatStr = "{used:.1f}{unit}",
    unit: MetricSymbol = 'G',
    *args, **kwargs
) -> Contents:
    '''
    System memory usage.

    :param fmt: A curly-brace format string with the following optional\
        named fields:
            - ``unit``: The same value as `unit`
            - ``available``: Available memory
            - ``total``: Total physical memory excluding swap
            - ``used``: Memory in use
            - ``percent``: Percent memory usage calculated as\
                ``(total - available) / total * 100``

        Defaults to ``"{free:.1f}{unit}"``
        See the `psutil documentation\
            <https://psutil.readthedocs.io/en/latest/#psutil\
            .virtual_memory>`_ for all possible optional named fields.
    :type fmt: :class:`FormatStr`

    :param unit: The unit prefix symbol, defaults to ``'G'``
    :type unit: :class:`MetricSymbol`

    :raises: :exc:`errors.InvalidArgError`
        if `unit` is not a valid :class:`MetricSymbol`
    '''
    if unit not in POWERS_OF_1024:
        raise InvalidArgError(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(POWERS_OF_1024)}."
        )

    memory = psutil.virtual_memory()
    factor = POWERS_OF_1024[unit]
    converted = {
        meas: (val if meas == 'percent' else val/factor)
        for meas, val in memory._asdict().items()
    }
    converted['unit'] = unit
    usage = fmt.format_map(converted)
    return usage


async def get_net_stats(
    # device: str = None,
    fmt: FormatStr = "{name}",
    nm: bool = True,
    nm_filt: NmConnFilterSpec = None,
    default: str = "",
    *args, **kwargs
) -> Contents:
    '''
    Active network info from either `NetworkManager` or `iwconfig`.

    :param fmt: A curly-brace format string with\
        five optional named fields:
            - ``name``: The connection name
            - ``uuid``: The connection uuid
            - ``type``: The connection type
            - ``device``: The connection device

        Defaults to ``"{name}"``
    :type fmt: :class:`FormatStr`

    :param nm: Use `NetworkManager`, defaults to ``True``
    :type nm: :class:`bool`

    :param nm_filt: Filter from active `NetworkManager` connections
    :type nm_filt: :class:`NmConnFilterSpec`, optional

    :param default: The string to replace `fmt` fields when there is no active connection.
    :type default: :class:`str`
    '''
    # If the user has NetworkManager, get the active connections:
    if nm:

        if nm_filt:
            # We use --terse for easier parsing.
            cmd = f"nmcli --terse connection show {shlex.join(nm_filt.items())}"
        else:
            cmd = "nmcli --terse connection show --active"

        proc = await aiosp.create_subprocess_shell(
            cmd,
            stdout=aiosp.PIPE,
            stderr=aiosp.PIPE
        )
        out, err = await proc.communicate()
        results = out.decode().splitlines()

        # Split in reverse to avoid butchering names with colons:
        conns = (c.rsplit(':', 3) for c in results)

        # By default, nmcli sorts connections by type:
        #   ethernet, wifi, tun, ...
        # We only need the first result.

        match next(conns, None):

            case None:
                keynames = ('name', 'uuid', 'typ', 'device')
                return fmt.format_map({key: default for key in keynames})

            case (name, uuid, typ, device):
                profile = {
                    # The command output duplicates all backslashes to
                    # escape them and adds more before colons in names.
                    # Take special care to preserve literal \ and : in
                    # original names while removing extra backslashes:
                    'name': name.replace('\\\\', '\\').replace('\\:', ':'),
                    'uuid': uuid,
                    'type': typ,
                    'device': device,
                }
                return fmt.format_map(profile)

            case _:
                return default

    else:
        cmd = await aiosp.create_subprocess_shell(
            "iwconfig",
            stdout=aiosp.PIPE
        )
        out, err = await cmd.communicate()
        if_list = out.decode().splitlines()

        ssid = next(
            line.split(':"')[1].strip('" ')
            for line in if_list
            if line.find("SSID")
        )
        return ssid


async def get_uptime(
    fmt: FormatStr,
    dynamic: bool = True,
    sep: str = ':',
    setupvars = None,
    *args, **kwargs
) -> Contents:
    '''
    System uptime.

    This function does some neat things.
    When `dynamic` is ``True``, the format string is split into groups
    using `sep`. If any format string fields in a group evaluate to
    0, the whole group is hidden.

    For example, using the format string ``"{days}d:{hours}h:{mins}m"``,
    86580 seconds is represented as ``1d:0h:3m`` when `dynamic` is
    ``False``, but would be shortened to ``1d:3m`` when `dynamic`
    is ``True``.

    Using only one unit, like ``"{hours}"``, will round `hours` to an
    integer by default. If floating point values are desired, a format
    spec must be used:
    ``"{hours:.4f}"``

    Format specs can be used to great effect, like in all other Field
    functions which accept `fmt`:
    ``"{hours:_^10.4f}"``

    .. seealso:: The Python `format spec mini language documentation
        <https://docs.python.org/3/library/string.html#format-\
        specification-mini-language>`_

    :param fmt: A curly-brace format string with seven optional named
        fields: ``secs``, ``mins``, ``hours``, ``days``, ``weeks``,
        ``months``, and ``years``

    :type fmt: :class:`FormatStr`

    :param dynamic: Given `sep`, automatically hide groups of units when
        they are ``0``, defaults to ``True``
    :type dynamic: :class:`bool`

    :param sep: Delimits groups of format fields to be hidden/shown
        together, defaults to ``":"``
    :type sep: :class:`str`
    '''
    secs = time.time() - psutil.boot_time()

    if not setupvars:
        conditional = ConditionalFormatStr(fmt, sep)

        setupvars = {
            'fnames': conditional.fnames,
            'groups': conditional.groups,
            'sep': sep,
        }
    lookup_table = ElapsedTime.in_desired_units(secs, setupvars['fnames'])
    setupvars['namespace'] = lookup_table

    if dynamic:
        out = format_uptime(secs, **setupvars)
        return out

    return fmt.format_map(setupvars['namespace'])

