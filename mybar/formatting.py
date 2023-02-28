from string import Formatter

from .errors import InvalidFormatStringFieldError
from .utils import join_options, make_error_message
from ._types import (
    Duration,
    FmtStrStructure,
    FormatterFname,
    FormatterConversion,
    FormatterFname,
    FormatterFormatSpec,
    FormatterLiteral,
    FormatStr,
)

from collections.abc import Callable, Iterable
from typing import Any, NamedTuple, Self


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


class FormatterFieldSig(NamedTuple):
    '''
    A format replacement field as generated by Formatter().parse().

    :param lit: Literal text preceding a replacement field
    :type lit: :class:`FormatterLiteral`

    :param name: The name of such a field, found inside curly braces
    :type name: :class:`FormatterFname`

    :param spec: The field's format spec, found inside curly braces
    :type spec: :class:`FormatterFormatSpec`

    :param conv: The field's conversion value, found inside curly braces
    :type conv: :class:`FormatterConversion`, optional
    '''

    lit: FormatterLiteral
    name: FormatterFname
    spec: FormatterFormatSpec
    conv: FormatterConversion

    @classmethod
    def from_str(cls, fmt: FormatStr) -> Self:
        '''
        Convert a replacement field to tuple of its elements.
        If there are multiple fields in the format string, only process the first one.

        :param fmt: The format string to convert
        :type fmt: :class:`FormatStr`
        '''
        try:
            parsed = tuple(Formatter().parse(fmt))

        except ValueError:
            err = f"Invalid format string: {fmt!r}"
            raise BrokenFormatStringError(err) from None

        if not parsed:
            err = f"The format string {fmt!r} contains no fields."
            raise FormatStringError(err)

        field = parsed[0]

        # Does the field have a fieldname?
        if field[1] == '':
            # No; it's positional.
            start = len(field[0])
            err = (
                f"The format string field at character {start} in {fmt!r} is "
                f"missing a fieldname.\n"
                 "Positional fields ('{}' for example) are not allowed "
                 "for this operation."
            )
            raise MissingFieldnameError(err)

        sig = cls(*field)
        return sig

    def as_string(self,
        with_literal: bool = True,
        with_conv: bool = True,
    ) -> FormatStr:
        '''
        Recreate a format string field from a single field signature.

        :param with_literal: Include the signature's :class:`FormatterLiteral`,
            defaults to ``True``
        :type with_literal: :class:`bool`

        :param with_conv: Include the signature's :class:`FormatterConversion`,
            defaults to ``True``
        :type with_conv: :class:`bool`

        :returns: The format string represented by the signature
        :rtype: :class:`FormatStr`
        '''
        inside_braces = self.name
        if with_conv and self.conv is not None:
            inside_braces += '!' + self.conv
        inside_braces += ':' + self.spec if self.spec else self.spec
        fmt = '{' + inside_braces + '}'
        if with_literal:
            return self.lit + fmt
        return fmt


class ConditionalFormatStr:
    '''
    Reinterpret format strings based on the data they reference.
    Values in a mapping which are predicated ``False`` have their
    groupings shown blank when the mapping is formatted.

    :param fmt: The initial format string
    :type fmt: :class:`FormatStr`

    :param sep: Surrounds related fields and literal text that should be
        grouped together, defaults to ``":"``
    '''
    def __init__(self,
        fmt: FormatStr,
        sep: str = ':'  # Other common values are , /
    ) -> None:
        self.fmt = fmt
        self.sep = sep
        self.fnames, self.groups = self.parse()

    def parse(self,
        sep: str = None
    ) -> tuple[tuple[FormatterFname], FmtStrStructure]:
        '''
        Parse a format string using its format fields and a separator.

        :param sep: Surrounds related fields and literal text that should be
            grouped together, defaults to ``self.sep``
        :type sep: :class:`str`, optional

        :returns: A nested tuple of the string's fieldnames
            and its :class:`FmtStrStructure`
        :rtype: :class:`tuple[tuple[FormatterFname], FmtStrStructure]`
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

        groups = tuple(
            tuple(Formatter().parse(section))
            for section in sections
        )

        fnames = tuple(
            name
            for section in groups
            for parsed in section
            if (name := parsed[1])
        )

        return fnames, groups

    def format(self,
        namespace: dict[FormatterFname, Any],
        predicate: Callable[[Any], bool] = bool,
        sep: str = None,
        substitute: str = None,
        round_by_default: bool = True,
    ) -> str:
        '''Format a dict of numbers according to a format string by parsing
        fields delineated by a separator `sep`.
        Field groups which fail the `predicate` are not shown in the
        final output string.
        If specified, `substitute` will replace invalid fields instead.

        :param namespace: A mapping with values to which fieldnames refer
        :type namespace: :class:`dict[FormatterFname]`

        :param predicate: A callable to determine whether a field should be
            printed, defaults to :func:`bool`
        :type predicate: :class:`Callable[[Any], bool]`

        :param sep: Used to join field groups into the final string,
            defaults to ``self.sep``
        :type sep: :class:`str`

        :param substitute: When set, replaces a field group that fails `predicate`
        :type substitute: :class:`str`, optional

        :param round_by_default: Round values if they are floats,
            defaults to ``True``
        :type round_by_default: :class:`bool`

        :returns: A string with field groups hidden which don't pass `predicate`
        :rtype: :class:`str`
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
    '''
    Represent elapsed time in seconds with a variety of larger units.
    '''
    conversions_to_secs = {
        'years': 12*4*7*24*60*60,
        'months': 4*7*24*60*60,
        'weeks': 7*24*60*60,
        'days': 24*60*60,
        'hours': 60*60,
        'mins': 60,
        'secs': 1,
        'femtofortnights': 14*24*60*60 * 10**-15  # You can't see this.
        }

    @classmethod
    def in_desired_units(cls,
        secs: int,
        units: tuple[Duration]
    ) -> dict[Duration, int]:
        '''
        Convert seconds to multiple units of time.

        The resulting value of the smallest unit in `units` is kept as
        a :class:`float`.
        Smaller units overflow into larger units when they meet or exceed
        the threshold given by
        :obj:`ElapsedTime.conversions_to_secs[larger_unit]`,
        but only if a larger unit is present in `units`.

        For example...

            - Running ``in_desired_units(12345, ('mins', 'hours'))``
                | yields ``{'hours': 3, 'mins': 25.75}``,
            - but ``in_desired_units(12345, ('mins', 'days'))``
                | yields ``{'days': 0, 'mins': 205.75}``,
            - and ``in_desired_units(12345, ('hours',))``
                | yields ``{'hours': 3.4291666666666667}``.

        :param secs: Seconds to be converted
        :type secs: :class:`int`

        :param units: Units to convert
        :type units: :class:`tuple[Duration]`

        :returns: A mapping of unit names to amount of time in those units
        :rtype: :class:`dict[Duration, int]`
        '''
        if not all(u in cls.conversions_to_secs for u in units):
            # At least one unit wasn't recognized. Raise an error:
            valid = cls.conversions_to_secs._safe()
            exptd = join_options(valid)
            unrec = join_options(set(units) - set(valid))
            exc = make_error_message(
                LookupError,
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

    class _Special_Obfuscating_Dict(dict):
        def _safe(self) -> Self:
            meme_hidden = self.copy()
            meme_hidden.pop('femtofortnights')  # Shhh, it's a secret.
            return meme_hidden

        def __repr__(self) -> str:
            return repr(self._safe())

    conversions_to_secs = _Special_Obfuscating_Dict(conversions_to_secs)
