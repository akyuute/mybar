__all__ = (
    'get_datetime',
    'get_hostname',
    'get_uptime',
    'get_cpu_usage',
    'get_cpu_temp',
    'get_mem_usage',
    'get_disk_usage',
    'get_battery_info',
    'get_net_stats',
    # 'get_audio_volume'
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
from .formatable import ElapsedTime, DynamicFormatStr
from .utils import join_options
from ._types import Contents, FieldName, FormatStr

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

NmConnIDSpecifier: TypeAlias = Literal['id', 'uuid', 'path', 'apath']
NmConnFilterSpec: TypeAlias = Iterable[NmConnIDSpecifier]

Duration: TypeAlias = Literal['secs', 'mins', 'hours', 'days', 'weeks']

POWERS_OF_1K = {
    'G': 3,
    'M': 2,
    'K': 1
}



##class Func:
##    f: Callable
##    s: Setup
##    pass

# Field functions

async def get_audio_volume(fmt="{:02.0f}{state}", *args, **kwargs) -> Contents:
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
    prec: int = 0, 
    fmt: str = "{:02.{}f}",
    *args,
    **kwargs
) -> Contents:
    '''Returns battery capacity as a percent and whether it is charging
    or discharging.'''

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
    info = fmt.format(battery.percent, prec)
    state = "CHG" if battery.power_plugged else ''
##    return battery.power_plugged, info
    return state + info

async def get_cpu_temp(
    fmt: str = "{:02.0f}{}",
    in_fahrenheit=False,
    *args,
    **kwargs
) -> Contents:
    '''Returns current CPU temperature in Celcius or Fahrenheit.'''
    symbols = ('C', 'F')
    symbol = symbols[in_fahrenheit]

    temps = psutil.sensors_temperatures(in_fahrenheit)
    match temps:
        case {'k10temp': t} | {'coretemp': t}:
            current = t[0].current
        case _:
            current = '??'
    return fmt.format(current, symbol)

async def get_cpu_usage(
    interval: float = None,
    fmt = "{:02.0f}%",
    *args,
    **kwargs
) -> Contents:
    '''Returns system CPU usage in percent over a specified interval.'''
    return fmt.format(psutil.cpu_percent(interval))

async def get_datetime(
    fmt: str = "%Y-%m-%d %H:%M:%S",
    *args,
    **kwargs
) -> Contents:
    '''Return the current time as formatted with `fmt`.'''
    return datetime.now().strftime(fmt)

def precision_datetime(
    fmt: str = "%Y-%m-%d %H:%M:%S.%f",
    *args,
    **kwargs
) -> Contents:
    '''Return the current time as formatted with `fmt`.
    Being synchronous, a threaded Field can run this with
    align_to_seconds and see less than a millisecond of offset.
    '''
    return datetime.now().strftime(fmt)

async def get_disk_usage(
    path='/',
    measure='free',
    unit='G',
    fmt: str = "{:.{}f}{}",
    prec: int = 1,
    *args,
    **kwargs
) -> Contents:
    '''Returns disk usage for a given path.
    Measure can be 'total', 'used', 'free', or 'percent'.
    Units can be 'G', 'M', or 'K'.'''

    if unit not in POWERS_OF_1K:
        raise InvalidArgError(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(POWERS_OF_1K, quote=True)}."
        )

    disk = psutil.disk_usage(path)
    if measure not in disk._fields:
        raise InvalidArgError(
            f"Invalid measure on this operating system: {measure!r}.\n"
            f"measure must be one of "
            f"{join_options(disk._fields, quote=True)}"
        )

    statistic = getattr(disk, measure, None)
    converted = statistic / 1024**POWERS_OF_1K[unit]
    usage = fmt.format(converted, prec, unit)
    return usage


async def get_hostname(*args, **kwargs) -> Contents:
    return os.uname().nodename

async def get_mem_usage(
    prec: int = 1,
    measure: str = 'used',
    unit: str = 'G',
    fmt: str = "{:.{}f}{}",
    *args,
    **kwargs
) -> Contents:
    '''Returns total RAM used including buffers and cache in GiB.'''

    if unit not in POWERS_OF_1K:
        raise InvalidArgError(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(POWERS_OF_1K, quote=True)}."
        )

    memory = psutil.virtual_memory()

    if measure not in memory._fields:
        raise InvalidArgError(
            f"Invalid measure on this operating system: {measure!r}\n"
            f"'measure' must be one of "
            f"{join_options(memory._fields, quote=True)}."
        )

    statistic = getattr(memory, measure, None)
    converted = statistic / 1024 ** POWERS_OF_1K[unit]
    mem = fmt.format(converted, prec, unit)
    return mem

#NOTE: This is most optimal as a threaded function.
async def get_net_stats(
    # device: str = None,
    nm: bool = True,
    nm_filt: NmConnFilterSpec = None,
    fmt: FormatStr = "{name}",
    default: str = "",
    *args,
    **kwargs
) -> Contents:
    '''Returns active network name (and later, stats) from either
    NetworkManager or iwconfig.
    '''
    # If the user has NetworkManager, get the active connections:
    if nm:

        if nm_filt:
            # We use --terse for easier parsing.
            cmd = f"nmcli --terse connection show {shlex.join(nm_filt)}"
        else:
            cmd = "nmcli --terse connection show --active"

        proc = await aiosp.create_subprocess_shell(
            cmd,
            stdout=aiosp.PIPE,
            stderr=aiosp.PIPE
        )
        out, err = await proc.communicate()
        results = out.decode().splitlines()

        conns = tuple(
            # Split in reverse to avoid butchering names with colons:
            c.rsplit(':', 3)
            for c in results
        )

        # Don't bother parsing an empty tuple:
        if not conns:
            keynames = ('name', 'uuid', 'typ', 'device')
            return fmt.format_map({key: default for key in keynames})

        # By default, nmcli sorts connections by type:
        #   ethernet, wifi, tun, ...
        # We only need the top result.
        match conns[0]:
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
                return ""

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

def get_uptime(
    *,
    fmt: FormatStr,
    dynamic: bool = True,
    sep: str = ':',
    setupvars = None,
) -> Contents:
    secs = round(time.time() - psutil.boot_time())

    if not setupvars:
        fnames, groups = DynamicFormatStr(fmt, sep).deconstruct()

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
    '''Fornat a dict of numbers according to a format string by parsing
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


