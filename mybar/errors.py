__all__ = (
    'BrokenFormatStringError',
    'CLIFatalError',
    'CLIUsageError',
    'CompatibilityWarning',
    'DefaultFieldNotFoundError',
    'FailedSetup',
    'FormatStringError',
    'IncompatibleArgsError',
    'InvalidArgError',
    'InvalidBarError',
    'InvalidFieldError',
    'InvalidFieldSpecError',
    'InvalidFormatStringFieldNameError',
    'InvalidOutputStreamError',
    'MissingFieldnameError',
    'UndefinedFieldError',
)


from os import PathLike
from typing import Any, IO
from .formatting import (
    FormatStringError,
    BrokenFormatStringError,
    InvalidFormatStringFieldNameError,
    InvalidFormatStringFormatSpecError,
    MissingFieldnameError
)
from ._types import Contents


class DefaultFieldNotFoundError(LookupError):
    '''
    Raised for references to an undefined default :class:`Field`.
    '''
    pass

class IncompatibleArgsError(ValueError):
    '''
    Raised when a class is instantiated with one or more missing or
    incompatible parameters.
    '''
    pass

class InvalidArgError(ValueError):
    '''
    Raised by a field function when it receives an invalid argument.
    '''
    pass

class InvalidFieldError(TypeError):
    '''
    Raised when a field is either not an instance of :class:`Field` or a string not
    found in the default fields container.
    '''
    pass

class InvalidFieldSpecError(TypeError):
    '''
    Raised when an expected :class:`FieldSpec` has the wrong type or an invalid structure.
    '''
    pass

class InvalidBarError(AttributeError):
    '''
    Raised when Field._check_bar() finds missing attributes in a
    potential status bar.
    '''

class InvalidOutputStreamError(AttributeError):
    '''
    Raised when an :class:`IO` stream lacks ``write()``,
    ``flush()`` and ``isatty()`` methods.
    '''
    pass

class UndefinedFieldError(LookupError):
    '''
    Raised if, when parsing a config file, a field name appears in the
    `field_order` item of the :class:`dict` passed to :meth:`Bar.from_dict()` that is neither
    found in its `field_definitions` parameter nor in :attr:`Field._default_fields`.
    '''
    pass


class CLIFatalError(Exception):
    '''
    Base class for errors that cause the CLI program to exit.

    :param msg: The error message to issue when exiting
    :type msg: :class:`str`
    '''
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg


class CLIUsageError(CLIFatalError):
    '''
    Raised when the CLI program is used incorrectly.
    '''
    pass


class FailedSetup(Exception):
    '''
    Raised by a setup function when it cannot return a suitable value.
    :meth:`Field.run` uses `backup` for the :class:`Bar` buffer value instead.

    :param backup: The field contents to use instead of the setup function return value
    :type backup: :class:`str`
    '''
    def __init__(self, backup: Contents) -> None:
        self.backup = backup


class CompatibilityWarning(Warning):
    '''
    Raised when certain application features will break if used in an
    environment without proper support.
    '''
