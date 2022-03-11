#!/usr/bin/python

"""Module docstring."""

import os
import re
import time
import psutil
import signal
import subprocess
from sys import stdout
from threading import Thread
from datetime import datetime as dt
from typing import Sequence, List, Tuple


Fields = {}  # Stores values for a StatusBar obj's fields which threads update
STOPTHREADS = False


class StatusThread(Thread):
    """Base class for status commands."""
    KEY = "?"  # Key for the global Fields dict; overwritten by subclasses
    INTERVAL = 1  # Interval at which threads are looped; subclasses override
    ISATTY = os.isatty(0)  # Indicates whether bar was run in a terminal or GUI
    ICON = ""  # Icons are unique to each subclass; subclasses override
    ICON_muted = ""  # Icon for when volume is muted; AudioVolume overrides
    ICON_dschrg = ""  # Icon for discharging battery; BatteryInfo overrides

    def __init__(self):
        Fields[self.KEY] = ""  # Update Fields dict using subclass' KEY
        super().__init__()  # Set up as threading.Thread object

    def get_value(self):
        """Returns a field value to be saved to the Fields dict.
        Ran by threads; optionally called by get_bar."""
        # raise NotImplemented("Needs to be overwritten by a subclass.")
        return

    def run(self):
        """Runs command threads much like StatusBar.run(),
        using unique INTERVAL instead of refresh_rate."""
        start = time.time()
        while not STOPTHREADS:
            Fields[self.KEY] = self.get_value()
            time.sleep(1)
            # time.sleep(1 - (time.time() - start) % 1)
            # time.sleep(self.INTERVAL - (time.time() - start) % self.INTERVAL)
         


class Hostname(StatusThread):
    KEY = "hostname"
    INTERVAL = 60
    ICON = ("", "")  # Use a tuple to index GUI/terminal icons with ISATTY

    def get_value(self):
        """Returns the system hostname."""
        hostname = os.uname().nodename  # This should work for UNIX and Windows
        return self.ICON[self.ISATTY] + hostname


class Uptime(StatusThread):
    KEY = "uptime"
    INTERVAL = 30
    ICON = (" ", "Up:")

    def get_value(self, *args, **kwargs):
        """Returns formatted system uptime in time since boot."""
        uptime = time.clock_gettime(time.CLOCK_BOOTTIME)

        def secs_to_dhm(n):
            """Converts seconds to a formatted time string."""
            mins, secs = divmod(n, 60)
            hours, mins = divmod(mins, 60)
            days, hours = divmod(hours, 24)
            if days:
                return "%dd:%dh:%dm" % (days, hours, mins)
            elif hours:
                return "%dh:%dm" % (hours, mins)
            elif mins:
                return "%dm" % mins
            else:
                return ""
        if secs_to_dhm(uptime) != "":
            return self.ICON[self.ISATTY] + secs_to_dhm(uptime)
        else:
            return ""


class CPUUsage(StatusThread):
    KEY = "cpu_usage"
    INTERVAL = 2
    ICON = (" ", "CPU:")

    def get_value(self, interval=INTERVAL, fmt: str = "02.0f", *args, **kwargs):
        """Returns system CPU usage in percent over a specified interval."""
        cpu_usage = f"{psutil.cpu_percent(interval):{fmt}}%"
        return self.ICON[self.ISATTY] + cpu_usage


class CPUTemp(StatusThread):
    KEY = "cpu_temp"
    INTERVAL = 2
    ICON = (" ", "Temp:")

    def get_value(self, in_fahrenheit=False, fmt: str = "02.0f", *args, **kwargs):
        """Returns current CPU temperature in Celcius or Fahrenheit."""
        coretemp = psutil.sensors_temperatures(in_fahrenheit)['coretemp']
        cpu_temp = f"""{
            next((temp.current for temp in coretemp if
                temp.label == "Package id 0"), "??"):{fmt}}{
            "F" if in_fahrenheit else "C"}"""
        # "Package id 0" is for the temp of the entire CPU
        return self.ICON[self.ISATTY] + cpu_temp


class MemUsage(StatusThread):
    KEY = "mem_used"
    INTERVAL = 2
    ICON = (" ", "Mem:")

    def get_value(self, prec=1, meas: str = "used", units: str = "G", fmt: str = ".1f", *args, **kwargs) -> str:
        """Returns total RAM used including buffers and cache in GiB."""
        memory = psutil.virtual_memory()
        sw_meas = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.total - memory.available,
            "active": memory.active,
            "inactive": memory.inactive,
            "buffers": memory.buffers,
            "cached": memory.cached,
            "shared": memory.shared
            }
        sw_units = {
            "G": (1024**3, "G"),
            "M": (1024**2, "M"),
            "K": (1024**1, "K")
        }

        try:
            used = f"{sw_meas[meas] / sw_units[units][0]:{fmt}}" + sw_units[units][1]
        except KeyError:
            raise KeyError(str(meas) +
                           "is not a valid psutil.virtual_memory() parameter.")
        return self.ICON[self.ISATTY] + used


class DiskUsage(StatusThread):
    KEY = "disk_usage"
    INTERVAL = 10
    ICON = (" ", "/:")

    def get_value(self, path="/", measure="free",
                  units="G", fmt: str = ".0f", *args, **kwargs):
        """Returns disk usage for a given path.
        Measure can be "total", "used", "free", or "percent".
        Units can be "G", "M", or "K"."""
        usage = psutil.disk_usage(path)
        sw_meas = {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": usage.percent
        }
        sw_units = {
            "G": (1024**3, "G"),
            "M": (1024**2, "M"),
            "K": (1024**1, "K")
        }
        result = sw_meas[measure] / sw_units[units][0]
        units_str = 3
        disk_usage = f"{result:{fmt}}{sw_units[units][1]}"
        return self.ICON[self.ISATTY] + disk_usage


class BatteryInfo(StatusThread):
    KEY = "battery_info"
    INTERVAL = 1
    ICON = (" ", "Chrg:")
    ICON_dschrg = (" ", "Bat:")
                                                            # Progressive/dynamic battery icons!


    # 
    # 
    # 
    # 
    # 


    def get_value(self, prec: int = None, *args, **kwargs):
        """Returns battery capacity as a percent and whether it is
        being charged or is discharging."""
        battery = psutil.sensors_battery()
        if not battery:
            return ""
        if battery.power_plugged:
            return self.ICON[self.ISATTY] + str(
                round(battery.percent, prec)).zfill(2)
        else:
            return self.ICON_dschrg[self.ISATTY] + str(
                round(battery.percent, prec)).zfill(2)


class NetStats(StatusThread):
    KEY = "net_stats"
    INTERVAL = 4
    ICON = (" ", "W:")

    def get_value(self, stats=False, nm=True):
        """Returns active network name (and later, stats) from either
        NetworkManager or iwconfig.
                                                                Note: add IO stats!
                                                                Using nmcli device show [ifname]:
                                                                Allow required device/interface argument, then:
                                                                Allow options for type, state, addresses, 

                                                                """
        def nm_ssid(stats):
            try:
                conns = subprocess.run(["nmcli", "con", "show", "--active"],
                                       timeout=1, capture_output=True,
                                       encoding="UTF-8").stdout.splitlines()
            except subprocess.TimeoutExpired as exc:
                # print("Command timed out:", exc.args[0])
                return ""
            for i in range(1, len(conns)):
                con = {k: v for k, v in zip(conns[0].split(), conns[i].split())}
                if con.get("TYPE") == "wifi":
                    con_name = str(con.get("NAME"))
                # elif con.get("TYPE") == "ether":
                    # con_name = "Ethernet"

                    return self.ICON[self.ISATTY] + con_name
            return ""

        def iw_ssid(stats):
            try:
                if_list = subprocess.run(["iwconfig"], timeout=1,
                    capture_output=True, encoding="ascii").stdout.splitlines()
            except subprocess.TimeoutExpired as exc:
                # print("Command timed out:", exc.args[0])
                return ""
            ssid = next(line.split(':"')[1].strip('" ') for line in if_list if line.find("SSID"))
            return self.ICON[self.ISATTY] + ssid

        if nm:
            return nm_ssid(stats)
        else: return iw_ssid(stats)


class AudioVolume(StatusThread):
    KEY = "audio_volume"
    # INTERVAL = 0.2
    INTERVAL = 3
    ICON = (" ", "Vol:")
    ICON_muted = ("", "Vol:0%")

    def get_value(self, *args, **kwargs):
        """Returns system audio volume from ALSA amixer. SIGUSR1 is used to
        notify the thread of volume changes from button presses."""
        try:
            output = subprocess.run(["amixer", "sget", "Master"],
                                timeout=1, capture_output=True,
                                encoding="UTF-8").stdout.splitlines()
        except subprocess.TimeoutExpired as exc:
            # print("Command timed out:", exc.args[0])
            return ""
        alsa_info = [
            line for line in output if line.find("Left:") != -1
            ][0].split(" ")
        if alsa_info[6] == "[0%]" or alsa_info[7] == "[off]":
            return self.ICON_muted[self.ISATTY]
        else:
            return self.ICON[self.ISATTY] + alsa_info[6].strip("][%").zfill(2)


class MusicInfo(StatusThread):
    KEY = "music_info"
    INTERVAL = 0.5
    ICON = (" ", "")

    def get_value(self, *args, **kwargs):
        """Returns formatted music info from MOC using mocp -Q command."""
        try:
            output = subprocess.run(["mocp", "-Q", "%state\n%file\n%tl"],
                                    timeout=1, capture_output=True,
                                    encoding="UTF-8").stdout.splitlines()
        except subprocess.TimeoutExpired as exc:
            # print("Command timed out:", exc.args[0])
            return ""
        if not output or output[0] == "STOP":
            return ""
        md = {k: v for k, v in zip(["state", "file", "timeleft"], output)}
        filepath = r"^/.+/"
        mocp_status = f"""[-{md["timeleft"]}]{">" if md["state"] == "PLAY"
            else "#"} {re.sub(filepath, "", md["file"])}"""
        return self.ICON[self.ISATTY] + mocp_status


class Datetime(StatusThread):
    KEY = "datetime"
    INTERVAL = 1
    ICON = ("", "")

    def get_value(self, fmt: str = "%Y-%m-%d %H:%M:%S"):
        return self.ICON[self.ISATTY] + dt.now().strftime(fmt)


# Instantiate the threads through their classes
Threads = [MusicInfo(), Hostname(), Uptime(), CPUUsage(),
           CPUTemp(), MemUsage(), DiskUsage(), BatteryInfo(),
           NetStats(), AudioVolume(), Datetime()]


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


def run_bar(refresh_rate, fields, sep, *args, **kwargs):
    """Runs the bar continuously and syncs it with the system clock."""
    for thread in Threads:
        if thread.KEY in fields:
            thread.start()

    if isinstance(sep, str):
        pass
    elif hasattr(sep, "__iter__"):
        sep = sep[os.isatty(0)]
    else:
        raise TypeError(
            "Separator must be of type str or list|tuple[str, str]")

    start = time.time()
    while True:
        bar = sep.join([Fields[field] for field in fields if Fields[field] != ""])
        stdout.flush()  # Flush Python buffer to stdout
        stdout.write(bar + "\r")
        for _ in range(int(10)):
            if STOPTHREADS:
                for thread in Threads:
                    thread.join()
        # Syncing the sleep timer to the system clock prevents drifting
        time.sleep(refresh_rate - (time.time() - start) % refresh_rate)



def main():

    # Create a dedicated /run/users/ and PID file directory for the current
    # process to which the volume update notifier can send a SIGUSR1
    PIDFILEDIR = os.path.join("/var/run/user", str(os.getuid()), "mybar")
    PID = os.getpid()
    PIDFILE = os.path.join(PIDFILEDIR, str(PID))

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
        Fields[AudioVolume.KEY] = AudioVolume().get_value()
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


    run_bar(fields=fields, refresh_rate=0.2, sep=(" | ", " "))
    # print(get_bar(fields=("cpu_usage"), sep=" | "))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        STOPTHREADS = True
        stdout.write("\n")
        exit(0)
