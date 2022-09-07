from string import Formatter
import psutil


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
##        raise InvalidArg(
##            f"Invalid unit: {unit!r}\n"
##            f"'unit' must be one of "
##            f"{join_options(UNITS, quote=True)}."
##        )
##
##    disk = psutil.disk_usage(path)
##    statistic = getattr(disk, measure, None)
##    if statistic is None:
##        raise InvalidArg(
##            f"Invalid measure on this operating system: {measure!r}.\n"
##            f"measure must be one of "
##            f"{join_options(statistic._fields, quote=True)}"
##        )
##
##    setupvars.update(
##        unit=unit,
##        measure=measure,
##    )

async def setup_uptime(
    # kwargs: dict,
    fmt: str,
    sep: str = None,
    *args,
    **kwargs
):
    setupvars = {}

##    if kwargs is None:
##        setupvars.update(
##            fnames=None,
##            deconstructed=None
##        )
##        return

    # sep = kwargs.get('sep')
    if sep is None:
        setupvars.update(
            dynamic=False,
        )
        return setupvars

    sections = fmt.split(sep)

    deconstructed = tuple(
        tuple(Formatter().parse(substr))
        for substr in sections
    )

    fnames = tuple(
        name
        for thing in deconstructed
        for tup in thing
        if (name := tup[1])
    )

    # return fnames, deconstructed
    setupvars.update(
        fnames=fnames,
        deconstructed=deconstructed
    )

    return setupvars

