#TODO: Docstrings!!
from string import Formatter

from .errors import InvalidFormatStringFieldError
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


class ConditionalFormatStr:
    '''
    '''
        #TODO: Docstring!!
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
        if sep is None:
            sep = self.sep

        sections = []
        # Split fmt for parsing, but join any format specs that get broken:
        pieces = (p for p in self.fmt.split(sep))

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
                doing_what=f"parsing {self!r} format string {self.fmt!r}",
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
        substitute: str = None,
        round_by_default: bool = True,
    ) -> str:
        '''Format a dict of numbers according to a format string by parsing
        fields delineated by a separator `sep`.
        Field groups which fail the `predicate` are not shown in the
        final format string.
        If specified, `substitute` will replace invalid fields instead.

        :param namespace: The namespace from which 
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
                # Handle sections that do not pass the predicate:
                if not predicate(val := namespace[maybe_field[1]]):

                    ##
                    # Maybe check the length to see if a regular .format() can be used!
                    ##

                    if substitute is not None:
                        newgroups.append(substitute)
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
                        elif round_by_default and isinstance(val, float):
                            buf += str(round(val))
                        else:
                            buf += str(val)

                    case weird:
                        raise ValueError(
                            f"\n"
                            f"Invalid structure in tuple\n"
                            f"  {i} {maybe_field}:\n"
                            f"  {weird!r}"
                        )

                if buf:
                    newgroup.append(buf)
            if newgroup:
                newgroups.append(newgroup)

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
        'secs': 1,
        'femtofortnights': 14*24*60*60 * 10**(-15)
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
            dememed = cls.conversions_to_secs.copy()
            dememed.pop('femtofortnights')
            exptd = join_options(dememed)
            valid = set(exptd)
            unrec = join_options(set(units) - valid)
            exc = make_error_message(
                LookupError,
                # InvalidFormatStringFieldError,
                # doing_what="finding units to convert",
                blame=repr(units),
                expected=f"a sequence of unit names from {exptd}",
                details=[
                    f"The following unit names are not recognized:",
                    f"{unrec}",
                ]
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
        # Give the least significant unit a precise value:
        last_u = ordered[-1]
        table[last_u] = secs / cls.conversions_to_secs[last_u]
        return table

