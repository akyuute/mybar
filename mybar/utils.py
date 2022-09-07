'''Utility functions'''

from typing import Iterable
from copy import deepcopy

__all__ = ('join_options', 'str_to_bool', 'make_error_message')

def join_options(
    it: Iterable[str],
    /,
    sep: str = ', ',
    final: str = ' or ',
    quote: bool = False,
    oxford: bool = False
):
    if not hasattr(it, '__iter__'):
        raise TypeError(f"Can only join an iterable, not {type(it)}.")
    opts = tuple(repr(e) if quote else e for e in it)
    return sep.join(opts[:-1]) + ('', ',')[oxford] + final + opts[-1]

def str_to_bool(value: str, /):
    '''Returns `True` or `False` bools for truthy or falsy strings.'''
    truthy = "true t yes y on 1".split()
    falsy = "false f no n off 0".split()
    pattern = value.lower()
    if pattern not in truthy + falsy:
        raise ValueError(f"Invalid argument: {value!r}")
    return (pattern in truthy or not pattern in falsy)

def clean_comment_keys(obj: dict, pattern: str = '//') -> dict:
    '''Returns a new dict with keys beginning with a comment pattern removed.'''
    # TODO: Support for obj: List!
    new = deepcopy(obj)
    for key, inner in tuple(new.items()):
        if key.startswith(pattern):
            new.pop(key)
        match inner:
            case str():
                if inner.startswith(pattern):
                    new.pop(key)
            case dict():
                clean_comment_keys(inner, pattern)
            case list():
                for i, foo in enumerate(inner):
                    if foo.startswith(pattern):
                        inner.pop(i)
    return new

def make_error_message(
    label: str,
    blame = None,
    expected: str = None,
    file: str = None,
    line: int = None,
    indent: str = "  ",
    details: Iterable[str] = None
) -> str:
    level = 0

    message = []
    if file is not None:
        message.append(f"In config file {file!r}")
        if line is not None:
            message[-1] += f" (line {line})"
        message[-1] += ":"
        level += 1

    elif line is not None:
        message.append(f"(line {line}):")
        level += 1

    message.append(f"{indent * level}While parsing {label}:")
    level += 1

    if details is not None:
        message.append('\n'.join((indent * level + det) for det in details))
        # message.append(
            # ('\n' + indent * level).join(details)
        # )

    if blame is not None:
        if expected is not None:
            message.append(
                f"{indent * level}Expected {expected}, "
                f"but got {blame} instead."
            )
        else:
            message.append(f"{indent * level}{blame}")

    err = ('\n').join(message)
    return err

