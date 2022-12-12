from os import PathLike
from typing import Any

class BrokenFormatStringError(ValueError):
    '''Raised when a format string cannot be properly parsed or contains
    positional fields ('{}').'''
    pass

class DefaultFieldNotFoundError(LookupError):
    '''Raised for references to an undefined default Field.'''
    pass

class IncompatibleArgsError(ValueError):
    '''Raised when a class is instantiated with one or more missing or
    incompatible parameters.'''
    pass

class InvalidArgError(ValueError):
    '''Raised by a field function when it receives an invalid argument.'''
    pass

class InvalidFieldError(TypeError):
    '''Raised when a field is either not an instance of Field or a string not
    found in the default fields container.'''
    pass

class InvalidFieldSpecError(TypeError):
    '''Raised when a Field specification mapping (processed by
    Bar.from_dict) has the wrong type or an invalid structure.
    '''
    pass

class InvalidFormatStringFieldError(LookupError):
    '''Raised when a format string has a field name not allowed or
    defined by kwargs in a str.format() call.'''
    pass

class InvalidOutputStreamError(AttributeError):
    '''Raised when an IO stream lacks write(), flush() and isatty() methods.'''
    pass

class MissingBarError(AttributeError):
    '''Raised when Field.run() is called before its instance is passed to the
    fields parameter in Bar().'''
    pass

class UndefinedFieldError(LookupError):
    '''Raised if, when parsing a config file, a field name appears in the
    'field_order' item of the dict passed to Bar.from_dict() that is neither
    found in its 'field_definitions' parameter nor in Field._default_fields.'''
    pass


class FatalError(Exception):
    '''Base class for errors that cause the CLI program to exit.'''
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

class CLIUsageError(FatalError):
    '''Raised when the CLI program is used incorrectly.'''
    pass


class AskWriteNewFile(Exception):
    '''Raised when Config.__init__ encounters a broken config file path.
    This allows the command line utility to ask the user if they would
    like the config file automatically written.
    '''
    def __init__(self, requested_file: PathLike) -> None:
        self.requested_file = requested_file

class FailedSetup(Exception):
    '''Raised when a setup function decides not to Field.run().
    The run() method uses args[0] for the buffer value as a backup.'''
    def __init__(self, backup: Any):
        self.backup = backup

