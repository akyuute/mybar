'''Utility functions'''

from copy import deepcopy

from ._types import FmtStrStructure, FormatStr, FormatterFname 

from collections.abc import Callable, Iterable
from typing import Any


def join_options(
    it: Iterable[object],
    /,
    sep: str = ', ',
    final_sep: str = 'or',
    quote: bool = True,
    oxford: bool = True,
    limit: int = None,
    overflow: str = '...',
    metasep: str = ' '
) -> str:
    '''
    Tie together a list of objects for use in natural sentences.
    Commonly used to present valid variable options or expected values
    in error messages.

    :param it: The iterable of objects to join
    :type it: :class:`Iterable`[``object``]

    :param sep: The string separating every option,
        defaults to ``', '``
    :type sep: :class:`str`

    :param final_sep: The string separating the last two options,
        defaults to ``'or'``
    :type final_sep: :class:`str`

    :param quote: Put each option in quotes,
        defaults to ``True``
    :type quote: :class:`bool`

    :param oxford: Put `sep` between second-last option and `final_sep`,
        otherwise `metasep`, defualts to ``True``
    :type oxford: :class:`bool`

    :param limit: Show this many options before appending `overflow`,
        defaults to ``None``
    :type limit: :class:`int`

    :param overflow: The string appended when `limit` options are joined,
        defaults to ``'...'``
    :type sep: :class:`str`

    :param metasep: Separates `final_sep` and the last option
    :type metasep: :class:`str`
    '''
    if not hasattr(it, '__iter__'):
        raise TypeError(f"Can only join an iterable, not {type(it)}.")
    match tuple(it):
        case ():
            return ""
        case (only,):
            return repr(str(only)) if quote else str(only)
    opts = [repr(str(item)) if quote else str(item) for item in it][:limit]
    if limit is not None and len(opts) >= limit:
        opts.append(overflow)
    elif oxford:
        opts[-1] = metasep.join((final_sep, opts[-1]))
    else:
        opts[-1] = metasep.join((opts.pop(-2), final_sep, opts[-1]))
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

