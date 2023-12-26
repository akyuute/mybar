__all__ = (
    'FormatterFieldSig',
    'FmtStrStructure',
    'ConditionalFormatStr',
    'ElapsedTime',
    'format_uptime',
)


from string import Formatter

from .utils import join_options, make_error_message
from ._types import (
    Duration,
    FormatStr,
    FormatterLiteral,
    FormatterFname,
    FormatterFormatSpec,
    FormatterConversion,
    Duration
)

from collections.abc import Callable, Container, Hashable, Iterable, Iterator
from typing import Any, NamedTuple, NoReturn, Self, TypeAlias


class FormatStringError(ValueError):
    '''
    Base class for exceptions involving format strings.
    '''
    pass

class BrokenFormatStringError(FormatStringError):
    '''
    Raised when a format string cannot be properly parsed or contains
    positional fields (``'{}'``).
    '''
    pass

class InvalidFormatStringFieldNameError(FormatStringError):
    '''
    Raised when a format string has a field name not allowed or
    not defined by kwargs in a :meth:`str.format` call.
    '''
    pass

class InvalidFormatStringFormatSpecError(FormatStringError):
    '''
    Raised when a format string has a field with a format spec
    incompatible with string values in a :meth:`str.format` call.
    '''
    pass


### Format String Wonderland ###

class FormatterFieldSig(NamedTuple):
    '''
    A NamedTuple representing a format replacement field as generated
    by Formatter().parse().

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.as_string()!r})"

    @classmethod
    def from_str(cls, fmt: FormatStr) -> Self:
        '''
        Convert a replacement field to a tuple of its elements.
        If there are multiple fields in the format string,
        only process the first one.

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

    def as_string(
        self,
        with_literal: bool = True,
        with_conv: bool = True,
        with_spec: bool = True
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
        if self.name is None:
            # This format string has no fields whatsoever.
            # But it may have a literal:
            if with_literal:
                return self.lit
            return ''
        # Otherwise, fields that exist but are empty have a name of ''.

        inside_braces = self.name
        if with_conv and self.conv:
            inside_braces += '!' + self.conv
        if with_spec and self.spec:
            inside_braces += ':' + self.spec
        fmt = '{' + inside_braces + '}'

        return self.lit + fmt if with_literal else fmt


class FmtStrStructure(tuple[FormatterFieldSig]):
    '''
    Represents the structure of a whole format string broken up by
    :meth:`string.Formatter.parse` into a tuple of
    :class:`FormatterFieldSig`, one for each replacement field.
    '''
    def __repr__(self) -> str:
        return type(self).__name__ + tuple.__repr__(self)

    def get_names(self) -> tuple[FormatterFname]:
        '''
        Return a tuple with all the format string's field names in order.
        A positional field will have a name of ``''``.
        '''
        names = (name for sig in self if (name := sig.name) is not None)
        # names = (name for sig in self if (name := sig.name))
        return tuple(names)

    @classmethod
    def from_str(cls, template: FormatStr) -> Self:
        try:
            sigs = tuple(
                FormatterFieldSig(*field)
                for field in Formatter().parse(template)
            )
        except ValueError:
            err = f"Invalid format string: {template!r}"
            raise BrokenFormatStringError(err) from None

        return cls(sigs)

    def validate_fields(
        self,
        valid_names: Container[FormatterFname] | None = None,
        check_positionals: bool = False,
        check_specs: bool = False,
    ) -> bool | NoReturn:
        '''
        Ensure the represented format string has valid replacement fields.

        :param valid_names: Provide a collection of valid names to ensure
            that each field name refers to a valid :class:`Field` object,
            defaults to None.
            Raise :exc:`InvalidFormatStringFieldNameError` for any fields
            with names not in `valid_names`.
            When `valid_names` is ``None``, field names are not checked.
        :type valid_names: Container[:class:`FormatterFname`]

        :param check_positionals: Ensure each field has a name,
            defaults to ``False``.
            Raise :exc:`MissingFieldnameError`
            if any field is missing a field name.
        :type check_positionals: :class:`bool`

        :param check_specs: Ensure the format spec in each field is valid,
            defaults to ``False``.
            Raise :exc:`InvalidFormatStringFormatSpecError` if any field has an
            invalid format spec.
        :type check_specs: :class:`bool`

        :returns: ``True`` if the format string passes these checks,
            and raises exceptions if not.
        :rtype: :class:`bool` | :class:`NoReturn`

        :raises: :exc:`errors.MissingFieldnameError` when `template`
            contains positional fields
        :raises: :exc:`InvalidFormatStringFieldNameError`
            When `valid_names` is ``None``, field names are not checked.
        :raises: :exc:`errors.InvalidFormatStringFormatSpecError` when
            `template` contains fields with invalid format specs
        '''
        if not (check_positionals or check_specs) or valid_names is None:
            return True

        if check_positionals:
            if any(sig.name == '' for sig in self):
                # There's a positional field somewhere.
                # Issue an actionable error message:
                epilogue = (
                    "Bar format string fields must all have fieldnames."
                    "\nPositional fields ('{}' for example) are invalid."
                )
                raise MissingFieldnameError.with_highlighting(self, epilogue)

        # Test that all format fields do their formatting correctly:
        for sig in self:
            # Skip sigs that only have a literal:
            if sig.name is None:
                continue

            if valid_names is not None:
                if sig.name not in valid_names:
                    err = make_error_message(
                        InvalidFormatStringFieldNameError,
                        blame=repr(sig.name),
                        expected=f"one of {join_options(valid_names)}",
                        details=(
                            f"{sig.name!r} is not an allowed"
                            f" format field name."
                            ,
                        ),
                    )
                    raise err from None

            if check_specs:

# This isn't testable in general, only for Fields...
##                # Ideally, check the function to which each field refers:
##                field = valid_names.get(sig.name, {})
##                func = field.get('func', None)
##                if func is None:
##                    rtype = 'unknown'
##                else:
##                    rtype = func.__annotations__['return']

                try:
                    # Just assume that the value will be a string:
                    format(str(), sig.spec)
                except ValueError:
                    invalid_spec_msg = (
                        f"{sig.spec!r} is not a valid format spec"
                        f" when formatting strings"
                        ,
                        f"with format field {sig.as_string()!r}."
                    )
                    err = make_error_message(
                        InvalidFormatStringFormatSpecError,
                        details=invalid_spec_msg,
                    )
                    raise err from None

        return True


class MissingFieldnameError(FormatStringError):
    '''
    Raised when a format string field lacks a fieldname (i.e. is positional)
    when one is expected.

    '''
    @classmethod
    def with_highlighting(
        cls,
        sigs: FmtStrStructure,
        epilogue: str = ''
    ) -> Self:
        '''
        Make an error message with highlighting under positional fields.

        :param sigs: The problematic format string in
            the form of a :class:`FmtStrStructure`
        :type sigs: :class:`FmtStrStructure`

        :param epilogue: A message at the end of the traceback
        :type epilogue: :class:`str`

        :returns: A new :class:`errors.MissingFieldnameError`
        :rtype: :class:`errors.MissingFieldnameError`
        '''
        rebuilt = ""
        highlight = " "  # Account for the repr quotation mark.

        for sig in sigs:
            field = sig.as_string()
            rebuilt += field

            if sig.name == '':  # For each positional field...
                # Skip over the part not in braces:
                highlight += " " * len(sig.lit)

                # Only highlight the part in braces.
                bad_field_len = len(field) - len(sig.lit)
                highlight += "^" * bad_field_len

            else:
                # Skip the whole field.
                highlight += " " * len(field)

        err = '\n'.join((
            "",
            "The following fields are missing fieldnames:",
            repr(rebuilt),
            highlight,
            epilogue
        ))

        return cls(err)


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
    YEARS = 12*4*7*24*60*60
    MONTHS = 4*7*24*60*60
    WEEKS = 7*24*60*60
    DAYS = 24*60*60
    HOURS = 60*60
    MINS = 60
    SECS = 1

    conversions_to_secs = {
        'years': YEARS,
        'months': MONTHS,
        'weeks': WEEKS,
        'days': DAYS,
        'hours': HOURS,
        'mins': MINS,
        'secs': SECS,
        'femtofortnights': 2 * WEEKS * 10**-15  # You can't see this.
    }
    '''
    A mapping of units of elapsed time to their equivalents in seconds.
    '''
    class dict(dict):
        '''
        Hide the silly unit.
        '''
        class SpecialStr(str):
            def __lt__(self, other) -> bool:
                '''
                Intercept sorted() (used by Sphinx).
                '''
                return False

        def __init__(self, *args, **kwargs) -> None:
            if args:
                args = ({self.SpecialStr(k): v for k, v in args[0].items()},)
            super().__init__(*args, **kwargs)

        def __iter__(self) -> Iterator:
            '''
            Intercept Sphinx.
            '''
            import inspect
            frame = inspect.getouterframes(inspect.currentframe())[1]
            if frame.function == 'object_description':
                self.pop('femtofortnights', None)  # Shhh, it's a secret.
            return super().__iter__()

        def __repr__(self) -> str:
            return repr(self._safe())

        def _safe(self) -> Self:
            hide_meme = self.copy()
            hide_meme.pop('femtofortnights', None)  # Shhh, it's a secret.
            return hide_meme

    conversions_to_secs = dict(conversions_to_secs)

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

        Running ``in_desired_units(12345, ('mins', 'hours'))``
             yields ``{'hours': 3, 'mins': 25.75}``,

        but ``in_desired_units(12345, ('mins', 'days'))``
             yields ``{'days': 0, 'mins': 205.75}``,

        and ``in_desired_units(12345, ('hours',))``
             yields ``{'hours': 3.4291666666666667}``.

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


def format_uptime(
    secs: int,
    sep: str,
    namespace: dict[Duration, int],
    groups: tuple[tuple[FormatterFieldSig]],
    *args,
    **kwargs
) -> str:
    '''
    Format a dict of numbers according to a format string by parsing
    fields delineated by a separator.

    :param secs: Total elapsed time in seconds (unused)
    :type secs: :class:`int`

    :param sep: A string that separates groups of text based on division
        of time
    :type sep: :class:`str`

    :param namespace: A mapping of time unit names to :class:`int`
    :type namespace: dict[:class:`Unit`, :class:`int`]

    :param groups: A format string broken up by
        :func:`_setups.setup_uptime` into tuples of
        :class:`FormatterFieldSig` based on the locations of `separator`
    :type groups: tuple[tuple[:class:`FormatterFieldSig`]]
    '''
    newgroups = []
    for i, group in enumerate(groups):
        if not group:
            # Just an extraneous separator.
            newgroups.append(())
            continue

        newgroup = []

        for maybe_field in group:
            # Skip groups that should appear blank:
            if (val := namespace.get(maybe_field[1])) is not None and val < 1:
                break

            buf = ""

            match maybe_field:
                case [lit, None, None, None]:
                    # A trailing literal.
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
                        try:
                            # Round down by default:
                            buf += str(int(val))
                        except TypeError:
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

    # Join everything.
    return sep.join(''.join(g) for g in newgroups)


##from typing import NamedTuple
### class Context(NamedTuple):
##    # value: str
##class Context(dict):
##    def __init__(
##        self,
##        *args,
##        value: str = None,
##    ) -> None: 
##        super().__init__(self)
##        self.value = value
##        # self.
##
##class IconFactory:
##    def __init__(
##        self,
##        func: Callable[[Context], Icon],
##        default: Icon = "",
##    ) -> None:
##        self.func = func
##        self.default = default
##
##    def interpret(self, ctx: Context = None, default: Icon = None) -> Icon:
##        if default is None:
##            default = self.default
##        if ctx is None:
##            return default
##        try:
##            return self.func(ctx)
##        except Exception:
##            return default
##
##
### class Icon:
### class IconMatcher(dict):
### class IconDict(dict):
##class EasyIcon(dict):
##
##    def __init__(
##        self,
##        map: dict[Hashable, Icon] = {},
##        default: Icon = "",
##        # picker = None
##    ) -> None:
##        # self._registry = statemap
##        if statemap:
##            self.update(statemap)
##        self.default = default
##        # if callable(picker):
##            # self.picker = picker
##        # return self
##
####    def __setitem__(self, state: Any, var: str) -> None:
####        self._registry[id(state)] = var
##        # self[id(state)] = var
##
####    def __getitem__(self, state: Any, ) -> str:
####        return self._registry.get(state)
##    def choose(self, obj: Any):
##        return self.get(obj, self.default)
##
##    def interpret(self, key: Any):
##        return self.get(self.picker(key), self.default)
##
##
##    def __repr__(self) -> str:
##        # return self._registry[self._default]
##        stuff = ', '.join(' on '.join((repr(v), repr(k))) for k, v in self.items())
##        cls = type(self).__name__
##        return f"{cls}({stuff}, default={self.default!r})"
##
##    # def __str__(self) -> str:
##        # return self._registry[self._default]
##
####cb = Callable[[Kwargs], State1|State2]
####result, state = cb()
####form = self.fmt.interpret(result, state)
####form = self.format(result, state)
##
##
##class Format:
##    content_key: str = '$'
##    def __init__(
##        self,
##        # statemap: dict = {},
##        icon="",
##        default_fmt: str = "{icon}{"+content_key+'}',
##        fallback: str = "",
##        # custom_does_it_all: Callable[['result', 'context'], str] = None
##    ) -> None:
##        # self.statemap = statemap
##        self.icon = icon
##        self.default_fmt = default_fmt
##        self.fallback = fallback
##
##
### class 
##class FormatSwitcher(dict):
### class FormatSwitcher(Format):
##    content_key: str = '$'
##    def __init__(self,
##        statemap: dict['result', str] = {},
##        default: str = "{"+content_key+"}",
##        only_switch_icons: bool = True,
##        fallback: str = "",
##    ) -> None:
##        # self.statemap = statemap
##        self.icon = icon
##        self.default = default
##        self.fallback = fallback
##        self.update(statemap)
##
##    def format(self, result) -> str:
##        if isinstance(result, str):
##            return self.get(result, self.defaul)
##
##        icon, fmt = self.statemap.get(context, (self.icon, self.default_fmt))
##        contents = fmt.format_map({self.content_key: result, 'icon': icon})
##        return contents
##
##
##class FormatMaker(Format):
##    def __init__(self,
##        fallback: str = "",
##        custom_does_it_all: Callable[['result'], str] = None
##    ) -> None:  
##        super().__init__(icon, default_fmt, fallback)
##
##        # if custom_does_it_all is None:
##            # custom_does_it_all = self._default_thingy
##
##    @staticmethod
##    def _default_thingy(fmt, result, context) -> str:
##        contents = fmt.format_map({content_key: result, 'icon': icon})
##        return
##
##    def format(self, result, context):
##        icon, fmt = self.statemap.get(context, (self.icon, self.default_fmt))
##        contents = fmt.format_map({self.content_key: result, 'icon': icon})
##        return contents
##
##def check_battery(ctx):
##    chrg_icn = ""
##    if ctx.get('charging'):
##        return chrg_icn + " "
##
##    icon_bank = """
##    
##    
##    
##    
##    
##    """.split()
##    def mapper(n):
##        icon = ""
##        for i, test in enumerate((10, 25, 50, 75, 100)):
##            if n <= test:
##                icon = icon_bank[i]
##                return icon + " "


