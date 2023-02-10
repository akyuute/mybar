from string import Formatter

from .utils import join_options, make_error_message
from ._types import Duration, FormatStr, FmtStrStructure, FormatterFname 

from collections.abc import Callable, Iterable
from typing import Any


class Icon(dict):
# class Icon:

    def __init__(
        self,
        statemap: dict[Any, str] = {},
        # default: 
    ) -> None:
        self._registry = statemap

        # return self

##    def __setitem__(self, state: Any, var: str) -> None:
##        self._registry[id(state)] = var
        # self[id(state)] = var

##    def __getitem__(self, state: Any, ) -> str:
##        return self._registry.get(state)

    def __repr__(self) -> str:
        # return self._registry[self._default]
        return ', '.join(' on '.join((repr(v), repr(k))) for k, v in self.items())

    # def __str__(self) -> str:
        # return self._registry[self._default]

### class Fmt(str):
##class Fmt:
##    pass

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
                doing_what=f"parsing {self!r} format string {fmt!r}",
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


class ElapsedTime:

    conversions_to_secs = {
        'years': 12*4*7*24*60*60,
        'months': 4*7*24*60*60,
        'weeks': 7*24*60*60,
        'days': 24*60*60,
        'hours': 60*60,
        'mins': 60,
        'secs': 1
    }

    @classmethod
    def in_desired_units(cls,
        secs: int,
        units: tuple[Duration]
    ) -> dict[Duration, int]:
        '''
        '''
        #TODO: Docstring!!

        if not all(u in cls.conversions_to_secs for u in units):
            exptd = join_options(cls.conversions_to_secs, quote=True)
            exc = make_error_message(
                KeyError,
                # doing_what="finding units to convert",
                blame=repr(units),
                expected=f"a sequence of units from {exptd}",
                details=[
                    f"One or more time unit names in {units!r} "
                    "are not recognized."]
            )
            raise exc

        # Get the units in order of largest first:
        ordered = tuple(u for u in cls.conversions_to_secs if u in units)

        table = {}
        if len(ordered) == 1:
            unit = ordered[0]
            # Avoid robbing the only unit of its precision. Just divide:
            table[unit] = secs / cls.conversions_to_secs[unit]
            return table

        for unit in ordered[:-1]:
            table[unit], secs = divmod(secs, cls.conversions_to_secs[unit])
        # table[unit] += secs  # Give the decimal back.  #NOTE THIS DON'T WORKKKKK
        table[ordered[-1]] = secs / cls.conversions_to_secs[ordered[-1]]  #NOTE Works. Pls simplify.
        return table

