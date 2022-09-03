import psutil
import re
import time

from asyncio import subprocess as aiosp
from datetime import datetime
from os import uname
from typing import Callable, Iterable

from .errors import *

__all__ = (
    'counter',
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

UNITS = {
    'G': 3,
    'M': 2,
    'K': 1
}


# Field functions

async def counter(cnt, *args, **kwargs):
    '''Your average garden variety counter.'''
    cnt[0] += 1
    return str(cnt[0])

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

    if unit not in UNITS:
        raise InvalidArg(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(UNITS, quote=True)}."
        )

    disk = psutil.disk_usage(path)
    statistic = getattr(disk, measure, None)
    if statistic is None:
        raise InvalidArg(
            f"Invalid measure on this operating system: {measure!r}.\n"
            f"measure must be one of "
            f"{join_options(statistic._fields, quote=True)}"
        )

    converted = statistic / 1024**UNITS[unit]
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

    if unit not in UNITS:
        raise InvalidArg(
            f"Invalid unit: {unit!r}\n"
            f"'unit' must be one of "
            f"{join_options(UNITS, quote=True)}."
        )

    memory = psutil.virtual_memory()
    statistic = getattr(memory, measure, None)

    if statistic is None:
        raise InvalidArg(
            f"Invalid measure on this operating system: {measure!r}\n"
            f"'measure' must be one of "
            f"{join_options(memory._fields, quote=True)}."
        )

    converted = statistic / 1024**UNITS[unit]
    mem = fmt.format(converted, prec, unit)
    return mem

async def get_net_stats(
    device=None,
    stats=False,
    nm=True,
    *args,
    **kwargs
):
    '''Returns active network name (and later, stats) from either
    NetworkManager or iwconfig.
    '''
    #NOTE: add IO stats!
        # Using nmcli device show [ifname]:
        # Allow required device/interface argument, then:
        # Allow options for type, state, addresses, 

    if nm:
        cmd = await aiosp.create_subprocess_shell(
            "nmcli con show --active",
            stdout=aiosp.PIPE
        )
        out, err = await cmd.communicate()
        if_list = out.decode().splitlines()

        # NOTE: This compresses whitespace:
        conns = (reversed(c.split()) for c in if_list)
        active_conns = (
            ' '.join(name)
            for device, typ, uuid, *name in conns
            if typ == 'wifi'
        )
        return next(active_conns, "")

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
    fmt: str = '%-jd:%-Hh:%-Mm',
    dynam_fmt = None,
    *args,
    **kwargs
):
    up_s = await uptime_s()
    timetuple = await secs_to_struct_t(up_s)
    return time.strftime(fmt, timetuple)

async def uptime_s():
    # return time.time() - psutil.boot_time()
    return (time.time_ns() // 1_000_000_000) - int(psutil.boot_time())

async def secs_to_struct_t(n: int):
    '''Returns a struct_time object useful for representing elapsed time.'''
##    '''A generator that counts down from n seconds, yielding the
##    remaining seconds and a struct_time object to represent them.'''
    limit = 367 * 24 * 60 * 60

    if n > limit:
        n = limit
    if n >= 0:
        mins, secs = divmod(n, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        timetuple = (0,0,0,hours,mins,secs,0,days,0)

        return time.struct_time(timetuple)
    return 0, time.struct_time((0,) * 9)

