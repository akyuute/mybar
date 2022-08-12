import asyncio
from asyncio import subprocess as aiosp
import os
import sys
import time
import psutil
import datetime
import inspect
from typing import Callable


async def get_datetime(fmt: str = "%Y-%m-%d %H:%M:%S", *_, **__):
    return datetime.datetime.now().strftime(fmt)

async def counter(cnt, *_, **__):
    cnt[0] += 1
    return str(cnt[0])

async def get_hostname(*_, **__):
    return os.uname().nodename

async def secs_to_struct_t(n: int):
    '''A generator that counts down from n seconds, yielding the
    remaining seconds and a struct_time object to represent them.'''
    # print("Making new remaining_t generator...")
    limit = 31708800
    # extra, n = divmod(s, limit)
    # print(f"{extra = }, {n = }, {s = }")
    # for it in range(extra + 1, 0, -1):
        # print(it)
    # n = s
    # if n <= limit:
    if n > limit:
        n = limit
    while n >= 0:
        # if s >= 31708800:
            # n -= 31708800
        mins, secs = divmod(n, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        timetuple = (0,0,0,hours,mins,secs,0,days,0)

        return n, time.struct_time(timetuple)

##        yield n, time.struct_time(timetuple)
        n -= 1
    # n = extra % limit
    # extra, n = divmod(n, limit)
    # print(f"{extra = }, {n = }, {s = }")

    return 0, time.struct_time((0,) * 9)

##    while True:
##        yield 0, time.struct_time((0,) * 9)


##def make_uptime_gen(fmt: str):
##    ups, up = secs_to_struct_t(uptime_s())
##    fmtd = time.strftime(fmt, up)
##    dynam_fmt = None
##    while True:
##        up = secs_to_struct_t(uptime_s())
##        fmtd = time.strftime(dynam_fmt(ups), up)
##        dynam_fmt = (yield fmtd) 


async def uptime_s(*_, **__):
# def uptime_s(*_, **__):
    # return time.time() - psutil.boot_time()
    return (time.time_ns() // 1_000_000_000) - int(psutil.boot_time())

async def get_uptime(fmt: str = '%-jd:%-Hh:%-Mm', *_, dynam_fmt = None, **__):
        # uptime = time.clock_gettime(time.CLOCK_BOOTTIME)
    # up_s = await uptime_s()
    up_s = await uptime_s()
    n, timetuple = await secs_to_struct_t(up_s)
    return time.strftime(fmt, timetuple)
##    while True:
##    up = secs_to_struct_t(uptime_s())
##    fmtd = time.strftime(fmt, up)
##    yield fmtd

##        def secs_to_dhm(n):
##            """Converts seconds to a formatted time string."""
##            mins, secs = divmod(n, 60)
##            hours, mins = divmod(mins, 60)
##            days, hours = divmod(hours, 24)
##            if days:
##                return "%dd:%dh:%dm:%ds" % (days, hours, mins, secs)
##            elif hours:
##                return "%dh:%dm:%ds" % (hours, mins, secs)
##            elif mins:
##                return "%dm:%ds" % (mins, secd)
##            else:
##                return None

        # return secs_to_dhm(uptime)

async def get_cpu_usage(
    interval: float = None,
    # fmt: str = None,
    fmt = "{:02.0f}%"
):
    # def get_value(self, interval=INTERVAL, fmt: str = "02.0f", *args, **kwargs):
        """Returns system CPU usage in percent over a specified interval."""
##        if fmt is None:
##            fmt = "{:02.0f}%"
        cpu_usage = fmt.format(psutil.cpu_percent(interval))
        return cpu_usage

async def get_cpu_temp(
    *,
    # fmt: str = None,
    fmt: str = "{:02.0f}{}",
    in_fahrenheit=False
):
    # def get_value(self, in_fahrenheit=False, fmt: str = "02.0f", *args, **kwargs):
    """Returns current CPU temperature in Celcius or Fahrenheit."""
    symbol = ('C', 'F')[in_fahrenheit]
##    if fmt is None:
##        fmt = "{:02.0f}{}"

    coretemp = psutil.sensors_temperatures(in_fahrenheit)['coretemp']  # Not universal!
    cpu_temp = next((temp.current for temp in coretemp if
    # "Package id 0" is for the temp of the entire CPU
        temp.label == "Package id 0"), "??")
    temp = fmt.format(cpu_temp, symbol)
    return temp


async def get_mem_usage(
    prec: int = 1,
    measure: str = "used",
    unit: str = "G",
    fmt: str = "{:.{}f}{}"
    # fmt: str = None,
):
    # """Returns total RAM used including buffers and cache in GiB."""
##    if fmt is None:
##        fmt = "{:.{}f}{}"

    switch_units = {
        "G": 3,
        "M": 2,
        "K": 1
    }

    if unit not in switch_units:
        err = (
            f"Invalid unit: {unit!r}.\n"
            f"unit must be one of {', '.join(repr(m) for m in switch_units)}"
        )
        raise KeyError(err)

    memory = psutil.virtual_memory()
    statistic = getattr(memory, measure, None)

    if statistic is None:
        err = (
            f"Invalid measure on this operating system: {measure=!r}.\n"
            f"measure must be one of {', '.join(repr(m) for m in memory._fields)}"
        )
        raise KeyError(err)
    converted = statistic / 1024**switch_units[unit]
    mem = fmt.format(converted, prec, unit)
    return mem

async def get_disk_usage(
    path="/",
    measure="free",
    unit="G",
    # fmt: str = None,
    fmt: str = "{:.{}f}{}",
    # prec: int = None,
    prec: int = 1,
):
    """Returns disk usage for a given path.
    Measure can be "total", "used", "free", or "percent".
    Units can be "G", "M", or "K"."""
##    if prec is None:
##        prec = 1
##    if fmt is None:
##        fmt = "{:.{}f}{}"

    switch_units = {
        "G": 3,
        "M": 2,
        "K": 1
    }

    if unit not in switch_units:
        err = (
            f"Invalid unit: {unit!r}.\n"
            f"unit must be one of {', '.join(repr(m) for m in switch_units)}"
        )
        raise KeyError(err)

    disk = psutil.disk_usage(path)
    statistic = getattr(disk, measure, None)

    if statistic is None:
        err = (
            f"Invalid measure on this operating system: {measure=!r}.\n"
            f"measure must be one of {', '.join(repr(m) for m in memory._fields)}"
        )
        raise KeyError(err)

    converted = statistic / 1024**switch_units[unit]
    usage = fmt.format(converted, prec, unit)
    return usage


# Progressive/dynamic battery icons!
    # 
    # 
    # 
    # 
    # 

async def get_battery_info(
    prec: int = 0, 
    # fmt: str = None,
    fmt: str = "{:02.{}f}"
):
    """Returns battery capacity as a percent and whether it is
    being charged or is discharging."""
    # if prec is None:
        # prec = 0
##    if fmt is None:
##        fmt = "{:02.{}f}"
    battery = psutil.sensors_battery()
    if not battery:
        return None
#         return (None, None)
    info = fmt.format(battery.percent, prec)
        # return self.ICON[self.ISATTY] + str(
            # round(battery.percent, prec)).zfill(2)
        # return self.ICON_dschrg[self.ISATTY] + str(
            # round(battery.percent, prec)).zfill(2)
#     return battery.power_plugged, info
    state = "CHG" if battery.power_plugged else ""
    return state + info

async def get_net_stats(device=None, stats=False, nm=True):
    """Returns active network name (and later, stats) from either
    NetworkManager or iwconfig.
    NOTE: add IO stats!
    Using nmcli device show [ifname]:
    Allow required device/interface argument, then:
    Allow options for type, state, addresses, 
    """

    if nm:
        cmd = await aiosp.create_subprocess_shell(
            "nmcli con show --active",
            # stdin=aiosp.PIPE,
            stdout=aiosp.PIPE,
        )
        out, err = await cmd.communicate()
        # if_list = out.splitlines()
        if_list = out.decode().splitlines()
        # print(f"{if_list = }")

##        # try:
##            keys, *out = subprocess.run(
##                "nmcli con show --active".split(),
##                timeout=1,
##                capture_output=True,
##                encoding='UTF-8'
##            ).stdout.splitlines()

##        except subprocess.TimeoutExpired as exc:
##            # print("Command timed out:", exc.args[0])
##            return None

        conns = (reversed(c.split()) for c in if_list)  # NOTE: This compresses whitespace
        active_conns = (
            ' '.join(name) for device, typ, uuid, *name in conns if typ == 'wifi'
        )
        # print(next(active_conns, "SOINSOC"))
        return next(active_conns, None)

    else:
        # try:
        cmd = await aiosp.create_subprocess_shell(
            "iwconfig",
            # stdin=aiosp.PIPE,
            stdout=aiosp.PIPE,
        )
        out, err = await cmd.communicate()
        if_list = out.decode().splitlines()
        # print(f"{if_list = }")
        # if_list = out.splitlines()

##        except subprocess.TimeoutExpired as exc:
##            # print("Command timed out:", exc.args[0])
##            return None

        ssid = next(line.split(':"')[1].strip('" ') for line in if_list if line.find("SSID"))
        # print(ssid)
        return ssid

async def get_audio_volume(*_, **__):
    """Returns system audio volume from ALSA amixer. SIGUSR1 is used to
    notify the thread of volume changes from button presses."""
    try:
        command = subprocess.run(
            ['amixer', 'sget', 'Master'],
            timeout=1,
            capture_output=True,
            encoding='UTF-8'
        )
    except subprocess.TimeoutExpired as exc:
        print("Command timed out:", exc.args[0])
        # return 'timed out'

    output = (l.strip() for l in reversed(command.stdout.splitlines()))

    pat = re.compile(r'.*\[(\d+)%\] \[(\w+)\]')
    pcts, states = zip(*(m.groups() for l in output if (m := pat.match(l))))
    avg_pct = sum(int(p) for p in pcts) // len(pcts)
    is_on = any(s == 'on' for s in states)
    return is_on, avg_pct


