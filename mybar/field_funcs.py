import psutil
import re
import shlex
import time

from asyncio import subprocess as aiosp
from datetime import datetime
from os import uname
from string import Formatter

from typing import Callable, Iterable
from mybar.custom_types import (
    FormatStr,
    NmConnIDSpecifier,
    Value
)

from mybar.errors import *

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
    'get_audio_volume'
)

POWERS_OF_1K = {
    'G': 3,
    'M': 2,
    'K': 1
}

DURATION_SWITCH = [
    ('mins', 'secs', 60),
    ('hours', 'mins', 60),
    ('days', 'hours', 24),
    ('weeks', 'days', 7),
]

ParserLiteral = str|None
ParserFname = str|None
ParserFormatSpec = str|None
ParserConversion = str|None
FieldStructure_T = tuple[tuple[tuple[
    ParserLiteral,
    ParserFname,
    ParserFormatSpec,
    ParserConversion
]]]


# Field functions

async def get_audio_volume(fmt="{:02.0f}{state}", *args, **kwargs):
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
):
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
):
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
):
    '''Returns system CPU usage in percent over a specified interval.'''
    return fmt.format(psutil.cpu_percent(interval))

async def get_datetime(fmt: str = "%Y-%m-%d %H:%M:%S", *args, **kwargs):
    return datetime.now().strftime(fmt)

async def get_disk_usage(
    path='/',
    measure='free',
    unit='G',
    fmt: str = "{:.{}f}{}",
    prec: int = 1,
    *args,
    **kwargs
):
    '''Returns disk usage for a given path.
    Measure can be 'total', 'used', 'free', or 'percent'.
    Units can be 'G', 'M', or 'K'.'''

    if unit not in POWERS_OF_1K:
        raise InvalidArg(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(POWERS_OF_1K, quote=True)}."
        )

    disk = psutil.disk_usage(path)
    statistic = getattr(disk, measure, None)
    if statistic is None:
        raise InvalidArg(
            f"Invalid measure on this operating system: {measure!r}.\n"
            f"measure must be one of "
            f"{join_options(statistic._fields, quote=True)}"
        )

    converted = statistic / 1024**POWERS_OF_1K[unit]
    usage = fmt.format(converted, prec, unit)
    return usage


async def get_hostname(*args, **kwargs):
    return uname().nodename

async def get_mem_usage(
    prec: int = 1,
    measure: str = 'used',
    unit: str = 'G',
    fmt: str = "{:.{}f}{}",
    *args,
    **kwargs
):
    '''Returns total RAM used including buffers and cache in GiB.'''

    if unit not in POWERS_OF_1K:
        raise InvalidArg(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(POWERS_OF_1K, quote=True)}."
        )

    memory = psutil.virtual_memory()
    statistic = getattr(memory, measure, None)

    if statistic is None:
        raise InvalidArg(
            f"Invalid measure on this operating system: {measure!r}\n"
            f"'measure' must be one of "
            f"{join_options(memory._fields, quote=True)}."
        )

    converted = statistic / 1024 ** POWERS_OF_1K[unit]
    mem = fmt.format(converted, prec, unit)
    return mem

async def get_net_stats(
    # device: str = None,
    nm: bool = True,
    # nm_filt: Sequence[str, str] = None,
    # nm_filt: NmConnectionFilterSpecifier = None,
    # nm_filt: Sequence[NmConnIDSpec, Value] = None,
    fmt: FormatStr = "{name}",
    default: str = "",
    *args,
    **kwargs
):
    '''Returns active network name (and later, stats) from either
    NetworkManager or iwconfig.
    '''
    # If the user has NetworkManager, get the active connections:
    if nm:

        # if nm_filt:
            # cmd = f"nmcli --terse connection show {shlex.join(nm_filt)}"
            # print(cmd)
        # else:
            # cmd = "nmcli --terse connection show --active"

        # Use --terse for easier parsing:
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

async def get_uptime(
    #TODO: Raise errors upon invalid field names!
    *,
    fmt: str,
    sep: str = None,
    dynamic: bool = True,
    reps: int = 4,
    fnames: Iterable[str] = None,
    deconstructed: Iterable = None,
):
    # n = time.time() - psutil.boot_time()
    n = int(time.time() - psutil.boot_time())
    lookup_table = await _secs_to_dhms_dict(n, reps)
    if not dynamic or sep is None:
        return fmt.format_map(lookup_table)

    out = await _format_numerical_fields(
        lookup_table=lookup_table,
        fmt=fmt,
        sep=sep,
        # **setupvars
        fnames=fnames,
        deconstructed=deconstructed,
        dynamic=dynamic,
    )
    return out

async def _secs_to_dhms_dict(n, reps):
    table = {'secs': n}
    # Get the field names in the right order:
    # indexes = tuple(keys.index(f) for f in fields)
    # granularity = keys[min(indexes)]
    # reps = max(indexes)
    for i in range(reps):
        fname, prev, mod = DURATION_SWITCH[i]
        table[fname], table[prev] = divmod(table[prev], mod)
    return table

async def _format_numerical_fields(
    lookup_table: dict[str, int|float],
    fmt: str,
    sep: str,

    # setup_uptime() supplies the following two args to avoid having to parse
    # the format string at each interval.

    # The list of format string field names:
    fnames: Iterable[str],
    # The nested tuple of deconstructed format string field data after going
    # through sep.split() and string.Formatter().parse():
    deconstructed: FieldStructure_T,
    dynamic: bool = True
) -> str:
    '''Fornat a dict of numbers according to a format string by parsing
    fields delineated by a separator.
    '''
    last_was_nonzero = True
    buffers = []
    for i, between_seps in enumerate(deconstructed):
        section = []
        
        for tup in between_seps:
            buf = ""

            match tup:
                case [lit, None, None, None]:
                    # A trailing literal.
                    # Only append if the previous field was valid.
                    if last_was_nonzero:
                        buf += lit

                case [lit, field, spec, conv]:
                    # A veritable field!
                    val = lookup_table.get(field)

                    # Should it be displayed?
                    if not val and dynamic:
                        last_was_nonzero = False
                        continue

                    # The text right before the field:
                    if lit is not None:
                        buf += lit

                    if spec:
                        buf += format(val, spec)
                    else:
                        buf += str(val)

                    last_was_nonzero = True

                case spam:
                    raise ValueError(
                        f"\n"
                        f"Invalid structure in tuple\n"
                        f"  {i} {tup}:\n"
                        f"  {spam!r}"
                    )

            if buf:
                section.append(buf)
        if section:
            buffers.append(section)

    # Join nested buffers.
    return sep.join(''.join(b) for b in buffers)

