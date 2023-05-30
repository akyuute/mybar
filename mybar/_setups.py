# import psutil

from .errors import (
    BrokenFormatStringError,
    FailedSetup,
    IncompatibleArgsError,
)
from .formatting import ConditionalFormatStr
from .utils import join_options, make_error_message 
from ._types import FormatStr

async def setup_uptime(
    fmt: FormatStr,
    sep: str = None,
    *args,
    **kwargs
) -> dict[str]:
    setupvars = {'fmt': fmt, 'sep': sep}
    try:
        conditional = ConditionalFormatStr(fmt, sep)
    except BrokenFormatStringError:
        raise FailedSetup(backup=fmt)
    else:
        setupvars.update({
            'fnames': conditional.fnames,
            'groups': conditional.groups,
        })
    return setupvars




##def setup_cpu_usage(
##    in_fahrenheit: bool = False,
##    *args,
##    **kwargs
##):
##    setupvars = {}
##
##    temps = psutil.sensors_temperatures(in_fahrenheit)
##    match temps:
##        case {'k10temp': t} | {'coretemp': t}:
##            current = t[0].current
##        case _:
##            current = '??'
##
##    return fmt.format(current, symbol)

##def setup_mem_usage(
##    prec: int = 1,
##    measure: str = 'used',
##    unit: str = 'G',
##    fmt: str = "{:.{}f}{}",
##    *args,
##    **kwargs
##):
##    setupvars = {}
##
##    if unit not in UNITS:
##        raise InvalidArgError(
##            f"Invalid unit: {unit!r}\n"
##            f"'unit' must be one of "
##            f"{join_options(UNITS, quote=True)}."
##        )
##
##    disk = psutil.disk_usage(path)
##    statistic = getattr(disk, measure, None)
##    if statistic is None:
##        raise InvalidArgError(
##            f"Invalid measure on this operating system: {measure!r}.\n"
##            f"measure must be one of "
##            f"{join_options(statistic._fields, quote=True)}"
##        )
##
##    setupvars.update(
##        unit=unit,
##        measure=measure,
##    )

