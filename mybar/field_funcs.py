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

import psutil

from .errors import *
from .formatable import ElapsedTime, ConditionalFormatStr
from .utils import join_options
from ._types import Contents, FormatStr, NmConnFilterSpec

from collections.abc import Callable, Iterable
from typing import Literal, TypeAlias


FormatterLiteral: TypeAlias = str|None
FormatterFname: TypeAlias = str|None
FormatterFormatSpec: TypeAlias = str|None
FormatterConversion: TypeAlias = str|None
FmtStrStructure: TypeAlias = tuple[tuple[tuple[
    FormatterLiteral,
    FormatterFname,
    FormatterFormatSpec,
    FormatterConversion
]]]

POWERS_OF_1024 = {
    'K': 1024**1,
    'M': 1024**2,
    'G': 1024**3,
    'T': 1024**4,
    'P': 1024**5,
}
MetricSymbol = Literal[*POWERS_OF_1024.keys()]

DiskMeasure = Literal['total', 'used', 'free', 'percent']


##class Func:
##    f: Callable
##    s: Setup
##    pass


# Field functions

async def get_audio_volume(
    fmt: FormatStr = "{:02.0f}{state}",
    *args, **kwargs
) -> Contents:
    '''Currently awaiting a more practical implementation that supports
    instantaneous updates. A socket would be quite nice for that.'''
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


async def get_battery_info(
    fmt: FormatStr = "{pct:02.0f}{state}",
    *args, **kwargs
) -> Contents:
    '''
    Battery capacity as a percent and whether or not it is charging.

    :param fmt: A curly-brace format string with
        two optional named fields:
            - ``pct``: Current battery percent as a :class:`float`
            - ``state``: Whether or not the battery is charging
        Defaults to ``"{pct:02.0f}{state}"``
    :type fmt: :class:`FormatStr`
    '''

    # Progressive/dynamic battery icons!
        # 
        # 
        # 
        # 
        # 

    # if not (battery := psutil.sensors_battery()):
    battery = psutil.sensors_battery()
    if not battery:
        return ""
##        # return (None, None)
    state = "CHG" if battery.power_plugged else ''
    info = fmt.format_map({'pct': battery.percent, 'state': state})
##    return battery.power_plugged, info
    return info


async def get_cpu_temp(
    fmt: str = "{temp:02.0f}{scale}",
    in_fahrenheit=False,
    *args, **kwargs
) -> Contents:
    '''
    Current CPU temperature in Celcius or Fahrenheit.

    :param fmt: A curly-brace format string with
        two optional named fields:
            - ``temp``: Current CPU temperature as a :class:`float`
            - ``scale``: ``'C'`` or ``'F'``, depending on `in_fahrenheit`
        Defaults to ``"{temp:02.0f}{scale}"``
    :type fmt: :class:`FormatStr`

    :param in_fahrenheit: Display the temperature in Fahrenheit instead of Celcius,
        defaults to ``False``
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


async def get_cpu_usage(
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


async def get_datetime(
    fmt: str = "%Y-%m-%d %H:%M:%S",
    *args, **kwargs
) -> Contents:
    '''
    Current time as formatted with `fmt`.

    :param fmt: A format string with %-based format codes used by ``datetime.strftime()``,
        defaults to ``"%Y-%m-%d %H:%M:%S"``
    :type fmt: :class:`str`

    .. seealso:: `strftime() format codes <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes>`_

    '''
    return datetime.now().strftime(fmt)


def precision_datetime(
    fmt: str = "%Y-%m-%d %H:%M:%S.%f",
    *args, **kwargs
) -> Contents:
    '''Return the current time as formatted with `fmt`.
    Being synchronous, a threaded Field can run this with
    align_to_seconds and see less than a millisecond of offset.
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

    :param fmt: A curly-brace format string with
        five optional named fields:
            - ``unit``: The same value as `unit`
            - ``total``: Total disk partition size
            - ``used``: Used disk space
            - ``free``: Free disk space (``total - used``)
            - ``percent``: Percent of total disk space that is used (``used / total``, a :class:`float`)
        Defaults to ``"{free:.1f}{unit}"``
    :type fmt: :class:`FormatStr`

    :param path: The path to a directory or file on a disk partition,
        defaults to ``'/'``
    :type path: :class:`os.PathLike`

    :param unit: The unit prefix symbol
    :type unit: :class:`MetricSymbol`
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


async def get_host(
    fmt: FormatStr = "{nodename}",
    *args, **kwargs
) -> Contents:
    '''
    System host information using :func:`os.uname()`.

    :param fmt: A curly-brace format string with
        five optional named fields:
            - ``nodename``: System hostname
            - ``sysname``: Operating system kernel name
            - ``release``: Kernel release
            - ``version``: Kernel version
            - ``machine``: Machine architecture
        Defaults to ``"{nodename}"``
    :type fmt: :class:`FormatStr`
    '''
    keys = ('sysname', 'nodename', 'release', 'version', 'machine')
    host = fmt.format_map(dict(zip(keys, os.uname())))
    return host


async def get_hostname(*args, **kwargs) -> Contents:
    '''System hostname.'''
    return os.uname().nodename


async def get_mem_usage(
    fmt: FormatStr = "{used:.1f}{unit}",
    unit: MetricSymbol = 'G',
    *args, **kwargs
) -> Contents:
    '''
    Disk usage of a partition at a given path.

    :param fmt: A curly-brace format string with
        five optional named fields:
            - ``unit``: The same value as `unit`
            - ``total``: Total disk partition size
            - ``used``: Used disk space
            - ``free``: ``total``-``used`` disk space
            - ``percent``: ``used``/``total`` disk space as a :class:`float`
        Defaults to ``"{free:.1f}{unit}"``
    :type fmt: :class:`FormatStr`

    :param path: The path to a directory or file on a disk partition,
        defaults to ``'/'``
    :type path: :class:`os.PathLike`

    :param unit: The unit prefix symbol
    :type unit: :class:`MetricSymbol`
    '''
    # '''Returns total RAM used including buffers and cache in GiB.'''
    memory = psutil.virtual_memory()
    factor = POWERS_OF_1024[unit]
    converted = {
        meas: (val if meas == 'percent' else val/factor)
        for meas, val in memory._asdict().items()
    }
    converted['unit'] = unit
    usage = fmt.format_map(converted)
    return usage


#NOTE: This is most optimal as a threaded function.
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

    :param fmt: A curly-brace format string with
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
        match next(conns):

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


# Uptime

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
    When `dynamic` is ``True``, the format string is split into groups using `sep`.
    If any format string fields in a group evaluate to 0, the whole group is hidden.
    For example, using the format string ``"{days}d:{hours}h:{mins}m"``, 86580 seconds is represented as ``"1d:0h:3m"`` with `dynamic` = ``False``, but would be shortened to ``"1d:3m"`` with `dynamic` = ``True``.
    ``"{hours}"`` will round `hours` to an integer by default.
    If :class:`float` values are desired, a format spec must be used:
        ``"{hours:.4f}"``
    Format specs can be used to great effect, like in all other field functions which accept `fmt`:
        ``"{hours:_^10.4f}"``

    :param fmt: A curly-brace format string with seven optional named fields:
        ``secs``, ``mins``, ``hours``, ``days``, ``weeks``, ``months``, and ``years``
    :type fmt: :class:`FormatStr`

    :param dynamic: Given `sep`, automatically hide groups of units when they are ``0``.
    :type dynamic: :class:`bool`

    :param sep: Delimits groups of format fields to be hidden/shown together,
        defaults to ``":"``
    :type sep: :class:`str`
    '''
    secs = time.time() - psutil.boot_time()

    if not setupvars:
        fnames, groups = ConditionalFormatStr(fmt, sep).deconstruct()

        setupvars = {
            'fnames': fnames,
            'groups': groups,
            'sep': sep,
        }
    lookup_table = ElapsedTime.in_desired_units(secs, setupvars['fnames'])
    setupvars['namespace'] = lookup_table

    if dynamic:
        out = format_uptime(secs, **setupvars)
        return out

    return fmt.format_map(setupvars['namespace'])


def format_uptime(
    secs: int,
    sep: str,
    namespace: dict[str],
    groups,
    *args, **kwargs
) -> str:
    '''Format a dict of numbers according to a format string by parsing
    fields delineated by a separator.
    '''
    newgroups = []
    for i, group in enumerate(groups):
        if not group:
            # Just an extraneous separator.
            newgroups.append(())
            continue

        newgroup = []
        
        for maybe_field in group:
            # Skip groups that should appear blank:
            if (val := namespace.get(maybe_field[1])
                ) == 0:
                break

            buf = ""

            match maybe_field:
                case [lit, None, None, None]:
                    # A trailing literal.
                    buf += lit

                case [lit, field, spec, conv]:
                    # A veritable format string field!
                    # Add the text right before the field:
                    if lit is not None:
                        buf += lit

                    # Format the value if necessary:
                    if spec:
                        buf += format(val, spec)
                    else:
                        try:
                            # Round floats by default:
                            buf += str(round(val))
                        except TypeError:
                            buf += str(val)

                case _:
                    raise ValueError(
                        f"\n"
                        f"Invalid structure in tuple\n"
                        f"  {i} {maybe_field}:\n"
                        f"  {spam!r}"
                    )

            if buf:
                newgroup.append(buf)
        if newgroup:
            newgroups.append(newgroup)

    # Join everything.
    return sep.join(''.join(g) for g in newgroups)

