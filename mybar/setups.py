import psutil
from string import Formatter

from mybar.errors import (
    BrokenFormatStringError,
    FailedSetup,
    IncompatibleArgsError,
    InvalidFormatStringFieldError,
)
from mybar.utils import join_options, make_error_message

FormatStr = str

FORMATTER = Formatter()

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

async def setup_uptime(
    # kwargs: dict,
    fmt: str,
    sep: str = None,
    *args,
    **kwargs
):
    setupvars = {}

    fnames = [name for tup in FORMATTER.parse(fmt) if (name := tup[1])]
    if not fnames:
        # Running the Field function is pointless if there are no
        # format string fields and the output is constant.
        # Exit gracefully using fmt as the value for constant_output:
        raise FailedSetup(fmt)

    durations = (
        'secs',
        'mins',
        'hours',
        'days',
        'weeks',
    )

    for name in fnames:
        if name not in durations:
            opts = join_options(durations, quote=True,)

            exc = make_error_message(
                InvalidFormatStringFieldError,
                doing_what="parsing get_uptime() format string",
                blame=name,
                expected=f"one of {opts}",
                details=[
                    f"Invalid get_uptime() format string field name: {name!r}"
                ]
            )
                    
            raise exc

    reps = max((durations.index(f) for f in fnames), default=0)

    setupvars.update(
        fnames=fnames,
        reps=reps
    )

    if not sep:
        setupvars.update(
            dynamic=False,
        )
        return setupvars

    # Split fmt for parsing, but join any format specs that get broken:
    sections = []
    pieces = (p for p in fmt.split(sep))

    try:
        for piece in pieces:
            while _is_malformed(piece):
                # Raise StopIteration if a valid field end is not found:
                piece = sep.join((piece, next(pieces)))
            sections.append(piece)

    except StopIteration:
        exc = make_error_message(
            BrokenFormatStringError,
            doing_what="parsing get_uptime() format string {fmt!r}",
            # blame=repr(piece),
            details=[
                f"Invalid fmt substring begins near ->{piece!r}"
                # f"{fmt[len(sep.join(sections)):]!r}:"
            ]
        )
        raise exc from None

    deconstructed = tuple(
        tuple(FORMATTER.parse(section))
        for section in sections
    )

    fnames = tuple(
        name
        for section in deconstructed
        for parsed in section
        if (name := parsed[1])
    )
    # print('\n', fnames)

    setupvars.update(
        deconstructed=deconstructed
    )
    # print("\n", setupvars, "\n")

    return setupvars

def _is_malformed(piece: FormatStr):
    '''Return whether piece is a malformed format string.'''
    try:
        tuple(FORMATTER.parse(piece))
    except ValueError:
        return True
    return False

