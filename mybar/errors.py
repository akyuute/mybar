class BrokenFormatString(ValueError):
    '''Raised when a format string cannot be properly parsed or contains
    positional fields ('{}').'''
    pass

class DefaultFieldNotFound(LookupError):
    '''Raised for references to an undefined default Field.'''
    pass

class FailedSetup(Exception):
    '''Raised when a setup function decides not to Field.run().
    The run() method uses args[0] for the buffer value as a backup.'''
    pass

class IncompatibleParams(ValueError):
    '''Raised when a class is instantiated with one or more missing or
    incompatible parameters.'''
    pass

class InvalidArg(ValueError):
    '''Raised by a field function when it receives an invalid argument.'''
    pass

class InvalidField(TypeError):
    '''Raised when a field is either not an instance of Field or a string not
    found in the default fields container.'''
    pass

class InvalidFieldSpec(TypeError):
    '''Raised when a Field specification mapping (processed by
    Bar.from_dict) has the wrong type or an invalid structure.
    '''
    pass

class InvalidOutputStream(AttributeError):
    '''Raised when an IO stream lacks write(), flush() and isatty() methods.'''
    pass

class MissingBar(AttributeError):
    '''Raised when Field.run() is called before its instance is passed to the
    fields parameter in Bar().'''
    pass

class UndefinedField(LookupError):
    '''Raised if, when parsing a config file, a field name appears in the
    'field_order' item of the dict passed to Bar.from_dict() that is neither
    found in its 'field_definitions' parameter nor in Field._default_fields.'''
    pass

class UndefinedFormatStringField(LookupError):
    '''Raised when a format string has a field name not allowed or
    defined by kwargs in a str.format() call.'''
    pass

