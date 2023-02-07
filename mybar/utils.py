'''Utility functions'''

from copy import deepcopy
from string import Formatter

from ._types import Duration, FmtStrStructure, FormatStr, FormatterFname 

from collections.abc import Callable, Iterable
from typing import Any


def join_options(
    it: Iterable[str],
    /,
    sep: str = ', ',
    final_sep: str = 'or ',
    quote: bool = False,
    oxford: bool = False,
    limit: int = None,
    overflow: str = '...',
) -> str:
    if not hasattr(it, '__iter__'):
        raise TypeError(f"Can only join an iterable, not {type(it)}.")
    opts = [repr(str(item)) if quote else str(item) for item in it][:limit]
    if limit is not None and len(opts) >= limit:
        opts.append(overflow)
    else:
        opts[-1] = final_sep + opts[-1]
    return sep.join(opts)


def str_to_bool(value: str, /) -> bool:
    '''Returns `True` or `False` bools for truthy or falsy strings.'''
    truthy = "true t yes y on 1".split()
    falsy = "false f no n off 0".split()
    pattern = value.lower()
    if pattern not in truthy + falsy:
        raise ValueError(f"Invalid argument: {value!r}")
    return (pattern in truthy or not pattern in falsy)


def recursive_scrub(
    obj: Iterable,
    /,
    test: Callable[[Any], bool],
    inplace: bool = False,
) -> Iterable:
    '''Scrub an iterable of any elements that pass a callable predicate.

    By default, return a scrubbed copy of the original object.
    For dicts, remove whole items whose keys pass the predicate.
    Remove elements recursively.
    '''
    new = obj
    if not inplace:
        new = deepcopy(obj)

    def clean(o):
        if isinstance(o, list):
            i = 0
            while i < len(o):
                elem = o[i]
                if test(elem):
                    del o[i]
                    continue
                else:
                    clean(elem)
                i += 1

        elif isinstance(o, dict):
            for key, val in tuple(o.items()):
                if test(key):
                    del o[key]
                # elif test(val):
                    # del o[key]
                else:
                    clean(val)

        elif test(o):
            del o

    clean(new)
    return new


def scrub_comments(
    obj: Iterable,
    /,
    pattern: str | tuple[str] = '//',
    inplace: bool = False
) -> Iterable:
    '''Scrub an iterable of any elements that begin with a substring.

    By default, return a scrubbed copy of the original object.
    For dicts, remove whole items whose keys match the pattern.
    Remove elements recursively.
    '''
    predicate = (lambda o: True if (
        isinstance(o, str) and o.startswith(pattern)
        ) else False
    )
    return recursive_scrub(obj, test=predicate, inplace=inplace)


def make_error_message(
    cls: Exception,
    doing_what: str = None,
    blame: Any = None,
    expected: str = None,
    details: Iterable[str] = None,
    epilogue: str = None,
    file: str = None,
    line: int = None,
    indent: str = "  ",
    indent_level: int = 0
) -> Exception:
    '''Dynamically build an error message from various bits of context.

    Return an exception with the message passed as args.
    '''
    level = indent_level

    message = []
    if file is not None:
        message.append(f"In file {file!r}")
        if line is not None:
            message[-1] += f" (line {line})"
        message[-1] += ":"
        level += 1

    if line is not None:
        message.append(f"(line {line}):")
        level += 1

    if doing_what is not None:
        message.append(f"{indent * level}While {doing_what}:")

    level += 1

    if blame is not None:
        if expected is not None:
            message.append(
                f"{indent * level}Expected {expected}, "
                f"but got {blame} instead."
            )
        else:
            message.append(f"{indent * level}{blame}")

    if details is not None:
        message.append(
            '\n'.join(
                (indent * level + det)
                for det in details
            )
        )
        # message.append(
            # ('\n' + indent * level).join(details)
        # )

    if epilogue is not None:
        # message.append(level * indent + epilogue)
        message.append(epilogue)

    err = '\n' + ('\n').join(message)
    return cls(err)


class ElapsedTime:
    @staticmethod
    def in_desired_units(
        units: tuple[Duration],
        secs: int
    ) -> dict[Duration, int]:

        mod_switch = {
            'years': 12*4*7*24*60*60,
            'months': 4*7*24*60*60,
            'weeks': 7*24*60*60,
            'days': 24*60*60,
            'hours': 60*60,
            'mins': 60,
            'secs': 1
        }

        # Get the units in order of largest first:
        ordered = (u for u in mod_switch if u in units)

        table = {}
        for unit in ordered:
            table[unit], secs = divmod(secs, mod_switch[unit])
        return table


class DynamicFormatStr:
    def __init__(self,
        fmt: FormatStr,
        sep: str = ':'  # Other common values are , /
    ) -> None:
        self.fmt = fmt
        self.sep = sep
        self.fnames, self.deconstructed = self.deconstruct()

    def deconstruct(self,
        sep: str = None
    ) -> tuple[tuple[FormatterFname], FmtStrStructure]:
        '''
        Break up a format string using its format fields and a separator.
        '''
        fmt = self.fmt
        if sep is None:
            sep = self.sep

        sections = []
        # Split fmt for parsing, but join any format specs that get broken:
        pieces = (p for p in fmt.split(sep))

        def _is_malformed(piece: FormatStr):
            '''Return whether a format string is malformed.'''
            try:
                tuple(Formatter().parse(piece))
            except ValueError:
                return True
            else:
                return False

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
                details=[
                    f"Invalid fmt substring begins near ->{piece!r}"
                ]
            )
            raise exc from None

        deconstructed = tuple(
            tuple(Formatter().parse(section))
            for section in sections
        )

        fnames = tuple(
            name
            for section in deconstructed
            for parsed in section
            if (name := parsed[1])
        )

        return fnames, deconstructed

    def format(self,
        namespace: dict[FormatterFname],
        predicate: Callable = bool,
        sep: str = None,
        replace: str = None,
    ) -> str:
        '''Fornat a dict of numbers according to a format string by parsing
        fields delineated by a separator `sep`.
        Field groups which fail the `predicate` are not shown in the
        final format string.
        If specified or ``None``, `replace` will be shown instead.
        '''
        if sep is None:
            sep = self.sep

        newgroups = []
        for i, group in enumerate(self.deconstructed):
            if not group:
                # Just an extraneous separator.
                newgroups.append(())
                continue

            newgroup = []
            
            for maybe_field in group:
                # Skip over sections that do not pass the predicate:
                if (val := namespace.get(maybe_field[1])
                    ) is not None and not predicate(val):
                    if replace is not None:
                        newgroups.append(replace)
                    break

                buf = ""

                match maybe_field:
                    case [lit, None, None, None]:
                        # A trailing literal.
                        # Only append if the previous field was valid:
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

        # return tuple(''.join(g) for g in newgroups)

        # Join everything.
        return sep.join(''.join(g) for g in newgroups)

