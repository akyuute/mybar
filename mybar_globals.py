#!/usr/bin/python

"""Module docstring."""

import os
import re
import time
import psutil
import signal
import pathlib
import subprocess
from sys import stdout
from string import Template
from threading import Thread
from datetime import datetime as dt
from typing import Sequence, List, Tuple

FIELDS = {}

PIDFILEDIR = os.path.join("/var/run/user", str(os.getuid()), "mybar")
PID = os.getpid()
PIDFILE = os.path.join(PIDFILEDIR, str(PID))

class Field:
    def __init__(
        self,
    ):
        pass


class Bar:
    # '''The main class for bars'''
    # Stores values for a StatusBar obj's fields which threads update:
    def __init__(self,
        field_names: list,
        refresh_rate: float = 1.0,
    ):
        self.fields = fields
        self.STOPTHREADS = False


class StatusThread(Thread):
    """Base class for status commands."""
    KEY = "?"  # Key for the global FIELDS dict; overwritten by subclasses
    INTERVAL = 1  # Interval at which threads are looped; subclasses override
    ISATTY = os.isatty(0)  # Indicates whether bar was run in a terminal or GUI
    ICON = ""  # Icons are unique to each subclass; subclasses override
    ICON_muted = ""  # Icon for when volume is muted; AudioVolume overrides
    ICON_dschrg = ""  # Icon for discharging battery; BatteryInfo overrides

    def __init__(self):
        FIELDS[self.KEY] = ""  # Update FIELDS dict using subclass' KEY
        super().__init__()  # Set up as threading.Thread object

    def get_value(self):
        """Returns a field value to be saved to the FIELDS dict.
        Ran by threads; optionally called by get_bar."""
        # raise NotImplemented("Needs to be overwritten by a subclass.")
        return

    def run(self):
        """Runs command threads much like StatusBar.run(),
        using unique INTERVAL instead of refresh_rate."""
        start = time.time()
        while not STOPTHREADS:
            FIELDS[self.KEY] = self.get_value()
            time.sleep(1)
            # time.sleep(1 - (time.time() - start) % 1)
            # time.sleep(self.INTERVAL - (time.time() - start) % self.INTERVAL)
         


## class Hostname(StatusThread):
##     KEY = "hostname"
##     INTERVAL = 60
##     ICON = ("", "")  # Use a tuple to index GUI/terminal icons with ISATTY

def get_hostname(interval: float, icon: list[str]):
    """Returns the system hostname."""
    hostname = os.uname().nodename  # This should work for UNIX and Windows
    return hostname
    # return self.ICON[self.ISATTY] + hostname


##class Uptime(StatusThread):
##    KEY = "uptime"
##    INTERVAL = 30
##    ICON = (" ", "Up:")


class StrfSecondsTemplate(Template):
    delimiter = '%'


def static_format_seconds(s: int, fmt: str):
    '''Converts seconds to a formatted time string.'''
    templ = StrfSecondsTemplate(fmt)
    mins, secs = divmod(s, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    d = dict(S=secs, M=mins, H=hours, D=days)
    return templ.substitute(d)
 

def dynamic_secs_to_dhm(s: int, /, *_):
    '''Converts seconds to a dhm-formatted time string,
    skipping fields with values == 0.'''
    mins, secs = divmod(s, 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    if days:
        return "%dd:%dh:%dm" % (days, hours, mins)
    elif hours:
        return "%dh:%dm" % (hours, mins)
    elif mins:
        return "%dm" % mins
    return ""


def get_uptime(formatter=None, fmt: str = None):
    # def get_value(self, *args, **kwargs):
        """Returns formatted system uptime in time since boot."""
        if formatter is None:
            formatter = dynamic_secs_to_dhm
        assert callable(formatter)
        if fmt is None:
            fmt = "%Dd:%Hh:%Mm"
        uptime = int(time.clock_gettime(time.CLOCK_BOOTTIME))
        # return uptime
        return formatter(uptime, fmt)


##class CPUUsage(StatusThread):
##    KEY = "cpu_usage"
##    INTERVAL = 2
##    ICON = (" ", "CPU:")

def get_cpu_usage(interval: float, fmt: str = None):
    # def get_value(self, interval=INTERVAL, fmt: str = "02.0f", *args, **kwargs):
        """Returns system CPU usage in percent over a specified interval."""
        if fmt is None:
            fmt = "{02.0f}%"
        cpu_usage = fmt.format(psutil.cpu_percent(interval))
        return cpu_usage
        # return self.ICON[self.ISATTY] + cpu_usage


##class CPUTemp(StatusThread):
##    KEY = "cpu_temp"
##    INTERVAL = 2
##    ICON = (" ", "Temp:")

def get_cpu_temp(*, fmt: str = None, in_fahrenheit=False):
    # def get_value(self, in_fahrenheit=False, fmt: str = "02.0f", *args, **kwargs):
    """Returns current CPU temperature in Celcius or Fahrenheit."""
    symbol = ('C', 'F')[in_fahrenheit]
    if fmt is None:
        fmt = "{:02.0f}{}"

    coretemp = psutil.sensors_temperatures(in_fahrenheit)['coretemp']
    cpu_temp = next((temp.current for temp in coretemp if
    # "Package id 0" is for the temp of the entire CPU
        temp.label == "Package id 0"), "??")
    temp = fmt.format(cpu_temp, symbol)
    return temp
    # return self.ICON[self.ISATTY] + cpu_temp


##class MemUsage(StatusThread):
##    KEY = "mem_used"
##    INTERVAL = 2
##    ICON = (" ", "Mem:")

def get_mem_usage(
    prec: int = 1,
    measure: str = "used",
    unit: str = "G",
    fmt: str = None,
):
    # """Returns total RAM used including buffers and cache in GiB."""
    if fmt is None:
        fmt = "{:.{}f}{}"

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
    # return self.ICON[self.ISATTY] + used
    converted = statistic / 1024**switch_units[unit]
    mem = fmt.format(converted, prec, unit)
    return mem


##class DiskUsage(StatusThread):
##    KEY = "disk_usage"
##    INTERVAL = 10
##    ICON = (" ", "/:")

def get_disk_usage(
    path="/",
    measure="free",
    unit="G",
    fmt: str = None,
    prec: int = None,
):
    """Returns disk usage for a given path.
    Measure can be "total", "used", "free", or "percent".
    Units can be "G", "M", or "K"."""
    if prec is None:
        prec = 1
    if fmt is None:
        fmt = "{:.{}f}{}"

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
    # return self.ICON[self.ISATTY] + disk_usage


##class BatteryInfo(StatusThread):
##    KEY = "battery_info"
##    INTERVAL = 1
##    ICON = (" ", "Chrg:")
##    ICON_dschrg = (" ", "Bat:")
                                                            # Progressive/dynamic battery icons!


    # 
    # 
    # 
    # 
    # 


def get_battery_info(prec: int = 0, fmt: str = None, ):
    """Returns battery capacity as a percent and whether it is
    being charged or is discharging."""
    # if prec is None:
        # prec = 0
    if fmt is None:
        fmt = "{:02.{}f}"
    battery = psutil.sensors_battery()
    if not battery:
        return (None, None)
    info = fmt.format(battery.percent, prec)
        # return self.ICON[self.ISATTY] + str(
            # round(battery.percent, prec)).zfill(2)
        # return self.ICON_dschrg[self.ISATTY] + str(
            # round(battery.percent, prec)).zfill(2)
    return battery.power_plugged, info


##class NetStats(StatusThread):
##    KEY = "net_stats"
##    INTERVAL = 4
##    ICON = (" ", "W:")


def get_net_stats(stats=False, nm=True):
    """Returns active network name (and later, stats) from either
    NetworkManager or iwconfig.
    NOTE: add IO stats!
    Using nmcli device show [ifname]:
    Allow required device/interface argument, then:
    Allow options for type, state, addresses, 
    """

    if nm:
        try:
            keys, *out = subprocess.run(
                "nmcli con show --active".split(),
                timeout=1,
                capture_output=True,
                encoding='UTF-8'
            ).stdout.splitlines()
        except subprocess.TimeoutExpired as exc:
            # print("Command timed out:", exc.args[0])
            return None

        conns = (reversed(c.split()) for c in out)  # NOTE: This compresses whitespace
        active_conns = (
            ' '.join(name) for device, typ, uuid, *name in conns if typ == 'wifi'
        )
        return next(active_conns, None)

    else:
        try:
            if_list = subprocess.run(["iwconfig"], timeout=1,
                capture_output=True, encoding="ascii").stdout.splitlines()
        except subprocess.TimeoutExpired as exc:
            # print("Command timed out:", exc.args[0])
            return None

        ssid = next(line.split(':"')[1].strip('" ') for line in if_list if line.find("SSID"))
        return ssid


##class AudioVolume(StatusThread):
##    KEY = "audio_volume"
##    # INTERVAL = 0.2
##    INTERVAL = 3
##    ICON = (" ", "Vol:")
##    ICON_muted = ("", "Vol:0%")

def get_audio_volume():
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


##class MusicInfo(StatusThread):
##    KEY = "music_info"
##    INTERVAL = 0.5
##    ICON = (" ", "")

def get_moc_status(fmt: str = None):
    """Returns formatted music info from MOC using mocp -Q command."""
    try:
        command = subprocess.run(
            ["mocp", "-Q", "%state\n%file\n%tl"],
            timeout=1,
            capture_output=True,
            encoding="UTF-8")
    except subprocess.TimeoutExpired as exc:
        print("Command timed out:", exc.args[0])
        # return ""

    output = command.stdout.splitlines()
    print(f"{output = }")

    if not output:
        return (None,) * 3
    # elif output[0] == "STOP":
    state, file, timeleft = output
        # return None
        # return ""
    # return state, file, timeleft

    filepath = pathlib.Path(file)
    filename = filepath.stem

    state_icon = '>' if state == 'PLAY' else '#'
    status = f'''[-{timeleft}]{state_icon} {filename}'''
    return status

##def moc_status_format(fmt: str = None,):
##    md = {k: v for k, v in zip(["state", "file", "timeleft"], output)}
##    filepath = r"^/.+/"
##    mocp_status = f"""[-{md["timeleft"]}]{">" if md["state"] == "PLAY"
##        else "#"} {re.sub(filepath, "", md["file"])}"""
##    return self.ICON[self.ISATTY] + mocp_status


##class Datetime(StatusThread):
##    KEY = "datetime"
##    INTERVAL = 1
##    ICON = ("", "")

##    def get_value(self, fmt: str = "%Y-%m-%d %H:%M:%S"):
##        return self.ICON[self.ISATTY] + dt.now().strftime(fmt)

def get_datetime(fmt: str = None):
    if fmt is None:
        fmt = "%Y-%m-%d %H:%M:%S"
    return time.strftime(fmt)


# Instantiate the threads through their classes
Threads = []
##        MusicInfo(), Hostname(), Uptime(), CPUUsage(),
##           CPUTemp(), MemUsage(), DiskUsage(), BatteryInfo(),
##           NetStats(), AudioVolume(), Datetime()]


def get_bar(fields, sep):
    if isinstance(sep, str):
        pass
    elif hasattr(sep, "__iter__"):
        sep = sep[os.isatty(0)]
    else:
        raise TypeError(
            "Separator must be of type str or list|tuple[str, str]")

    bar_list = [thread.get_value() for thread in Threads
                if thread.KEY in fields and thread.get_value()]
    return sep.join(bar_list)


def run_bar(
    fields,
    refresh_rate: float,
    sep: str,
    *args, **kwargs):
    """Runs the bar continuously and syncs it with the system clock."""
    if isinstance(sep, str):
        sep = sep
    elif hasattr(sep, "__iter__"):
        sep = sep[os.isatty(0)]
    else:
        raise TypeError(
            "Separator must be of type str or list|tuple[str, str]")

    try:
        for thread in Threads:
            if thread.KEY in fields:
                thread.start()

        start = time.time()

        while not STOPTHREADS:
            bar = sep.join([FIELDS[field] for field in fields if FIELDS[field] != ""])
            stdout.flush()  # Flush Python buffer to stdout
            stdout.write(bar + "\r")
            for _ in range(int(10)):
                if STOPTHREADS:
                    for thread in Threads:
                        thread.join()
            # Syncing the sleep timer to the system clock prevents drifting
            time.sleep(refresh_rate - (time.time() - start) % refresh_rate)
    except KeyboardInterrupt:
        STOPTHREADS = True
        stdout.write("\n")
        exit(0)




def main():

    # Create a dedicated /run/users/ and PID file directory for the current
    # process to which the volume update notifier can send a SIGUSR1
    if not os.isatty(0):
        try:
            # Remove any old PID files and create a new directory if necessary
            for file in os.listdir(PIDFILEDIR):
                os.remove(os.path.join(PIDFILEDIR, file))
        except (FileNotFoundError, NotADirectoryError):
           os.makedirs(PIDFILEDIR)

    # Write or overwrite PID file, only if in a GUI
        with open(PIDFILE, "w") as f:
            f.write(str(PID))

    # Using an external tool such as a shell script, have the volume buttons
    # send a SIGUSR1 signal to notify audio thread of the change
    # Handle the signal:
    def receive_update_sig(signum, frame):
        # Update audio display value when volume changes and signal is received
        FIELDS[AudioVolume.KEY] = AudioVolume().get_value()
    # Register the signal handler
    signal.signal(signal.SIGUSR1, receive_update_sig)


    fields = (
        "music_info",
        "uptime",
        "cpu_usage",
        "cpu_temp",
        "mem_used",
        "disk_usage",
        "battery_info",
        "net_stats",
        "audio_volume",
        "datetime"
        )


    # run_bar(fields=fields, refresh_rate=0.2, sep=("|", " "))
    # print(get_bar(fields=("cpu_usage"), sep=" | "))


if __name__ == "__main__":
    pass
    try:
        main()
    except KeyboardInterrupt:
        STOPTHREADS = True
        stdout.write("\n")
        exit(0)
