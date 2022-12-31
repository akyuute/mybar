from os import PathLike
from typing import Any, IO

from .types import Contents, FieldSpec


class BrokenFormatStringError(ValueError):
    '''Raised when a format string cannot be properly parsed or contains
    positional fields (``'{}'``).'''
    pass

class DefaultFieldNotFoundError(LookupError):
    '''Raised for references to an undefined default :class:`Field`.'''
    pass

class IncompatibleArgsError(ValueError):
    '''Raised when a class is instantiated with one or more missing or
    incompatible parameters.'''
    pass

class InvalidArgError(ValueError):
    '''Raised by a field function when it receives an invalid argument.'''
    pass

class InvalidFieldError(TypeError):
    '''Raised when a field is either not an instance of :class:`Field` or a string not
    found in the default fields container.'''
    pass

class InvalidFieldSpecError(TypeError):
    '''Raised when an expected :class:`FieldSpec` has the wrong type or an invalid structure.
    '''
    pass

class InvalidFormatStringFieldError(LookupError):
    '''Raised when a format string has a field name not allowed or
    not defined by kwargs in a :meth:`str.format()` call.'''
    pass

class InvalidOutputStreamError(AttributeError):
    '''Raised when an :class:`IO` stream lacks ``write()``,
    ``flush()`` and ``isatty()`` methods.'''
    pass

class MissingBarError(AttributeError):
    '''Raised when :meth:`Field.run()` is called before its instance is passed to the
    `fields` parameter in :meth:`Bar.__init__()`.'''
    pass

class UndefinedFieldError(LookupError):
    '''Raised if, when parsing a config file, a field name appears in the
    `field_order` item of the :class:`dict` passed to :meth:`Bar.from_dict()` that is neither
    found in its `field_definitions` parameter nor in :attr:`Field._default_fields`.'''
    pass


class FatalError(Exception):
    '''
    Base class for errors that cause the CLI program to exit.

    :param msg: The error message to issue when exiting
    :type msg: :class:`str`
    '''
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

class CLIUsageError(FatalError):
    '''Raised when the CLI program is used incorrectly.'''
    pass


class AskWriteNewFile(Exception):
    '''
    Raised when :meth:`Template.from_file` is given a broken config file path.
    This allows the command line utility to ask the user if they would
    like the config file automatically written.

    :param requested_file: The nonexistent file requested by user input
    :type requested_file: :class:`PathLike`
    '''
    def __init__(self, requested_file: PathLike) -> None:
        self.requested_file = requested_file


class FailedSetup(Exception):
    '''
    Raised by a setup function when it cannot return a suitable value.
    :meth:`Field.run` uses `backup` for the :class:`Bar` buffer value instead.

    :param backup: The field contents to use instead of the setup function return value
    :type backup: :class:`str`
    '''
    def __init__(self, backup: Contents) -> None:
        self.backup = backup

