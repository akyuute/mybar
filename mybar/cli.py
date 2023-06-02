__all__ = (
    'Parser',
    'OptionsAsker',
)


from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
from enum import Enum

from .errors import AskWriteNewFile, CLIFatalError, CLIUsageError
from .templates import BarConfigSpec, FieldSpec
from ._types import (
    AssignmentOption,
    FieldName,
    OptName,
    OptSpec,
)

from typing import Any, Callable, Iterable, NoReturn


PROG = __package__


class ArgFormatter:
    '''
    Methods for formatting args.
    '''
    @staticmethod
    def SplitFirst(char: str) -> Callable[[str], str]:
        '''
        Give this to the `type` parameter of :func:`ArgumentParser.add_argument`
        and it will split args using `char`.
        '''
        return (lambda f: f.split(char, 1))

    @staticmethod
    def ToTuple(length: int) -> Callable[[Any], tuple]:
        '''
        Give this to the `type` parameter of :func:`ArgumentParser.add_argument`
        and it will fill tuples with `length` copies of the arg.
        '''
        return (lambda s: (s,) * length)


class Parser(ArgumentParser):
    '''A custom command line parser used by the command line utility.'''
    assignment_arg_map = {
        'icon_pairs': 'field_icons',
    }

    def __init__(self) -> None:
        super().__init__(
            prog=PROG,
            argument_default=SUPPRESS,
        )
        self.add_arguments()

    def parse_args(self, args: None | list[str] = None) -> BarConfigSpec:
        '''
        Parse command line arguments and return a dict of options.
        This will additionally process args with key-value pairs.
        If `args` is None, process from STDIN.

        :param args: The args to parse, get from STDIN by default
        :type args: :class:`list[str]`
        '''
        # Use vars() because dict items are more portable than attrs.
        opts = vars(super().parse_args(args))
        params = self.process_assignment_args(opts)
        return params

    def process_assignment_args(
        self,
        opts: BarConfigSpec,
        assignments: dict[FieldName, str] = None
    ) -> BarConfigSpec:
        '''
        Make dicts from key-value pairs in assignment args.
        How does it work?
        I don't remember!
        '''
        if assignments is None:
            assignments = self.assignment_arg_map

        for src, new_dest in assignments.items():
            if (vals := opts.pop(src, None)):
                opts[new_dest] = dict(
                    pair for item in vals
                    # Only use items that are pairs:
                    if len(pair := item.split('=', 1)) == 2
                )
        return opts

    @staticmethod
    def process_field_options(
        options: Iterable[AssignmentOption]
    ) -> dict[FieldName, FieldSpec]:
        '''
        Parse variable assignment args given for Field definitions.
        Each option in `options` should look like 'key=val'.
        We specifically handle Field options. To that end, parse nested
        options with emulated dotted attribute access by ignoring all
        options not of the form 'field.key=val'.
        After parsing options, return a dict mapping them to Field names,
        similar to `field_definitions` in :meth:`Bar.from_config()`.

        I think this has some logical chinks. Definitely overengineered.

        :param options: Assignment option 'key=val' pairs to parse
        :type options: Iterable[:class:`AssignmentOption`]

        :returns: a dict mapping Field names to dicts of options
        :rtype: :class:`dict`[:class:`FieldName`, :class:`FieldSpec`]
        '''
        field_definitions = {}
        for opt in options:
            match (pair := opt.split('=', 1)):

                case [positional]:
                    # Looks like '--opt'
                    # Skip command options.
                    continue

                case [field_and_opt, val]:  # Looks like 'key=val'
                    if field_and_opt.isidentifier():
                        # No dots means it's an option for the Bar.
                        # Not our problem.
                        continue

                    if val == "''":  # Looks like "key=''"
                        val = ''
                    elif not val:  # Looks like 'key='
                        val = None

                    # Looks like 'field.key=val'
                    # An option for a Field.
                    # Handle attribute access through dots:
                    field_name, field_opt = field_and_opt.split('.', 1)
                    if field_name not in field_definitions:
                        field_definitions[field_name] = {}

                    maybe_kwargs = field_opt.split('.', 1)

                    match maybe_kwargs:

                        case ['kwargs']:
                            # Looks like 'field.kwargs={"k": "v", ...}'
                            val = {} if val is None else eval(val)
                            # Yes, I know, it's eval.

                        case ['kwargs', nested]:
                            if nested.isidentifier():
                                # Looks like 'field.kwargs.k=v'
                                kwarg = {nested: val}
                                val = kwarg
                                field_opt = 'kwargs'

                            else:
                                pass

                        case [not_kwargs]:
                            # field_opt is valid.
                            pass

                        case _:
                            # Nonsense
                            msg = (
                                f"{opt !r} is invalid option syntax."
                            )
                            raise CLIUsageError(msg)

                    field_definitions[field_name].update({field_opt: val})

        return field_definitions

    def add_arguments(self) -> None:
        '''Equip the parser with all its arguments.'''
        fields_or_fmt = self.add_mutually_exclusive_group()
        fields_or_fmt.add_argument(
            '-t', '--template',
            metavar="'TEMPLATE'",
            dest='template',
            help=(
                "A curly-brace-delimited format string. "
                "Not valid with --fields/-f options."
            ),
        )

        fields_or_fmt.add_argument(
            '-f', '--fields',
            action='extend',
            nargs='+',
            metavar=('FIELDNAME1', 'FIELDNAME2'),
            dest='field_order',
            # type=ArgFormatter.SplitFirst(','),
            help=(
                "A list of fields to be displayed. "
                "Not valid with --format/-m options."
            ),
        )

        fields_group = self.add_argument_group(
            title="Options for fields",
            description="These options are not valid when using --format/-m."
        )

        fields_group.add_argument(
            '--icons',
            action='extend',
            nargs='+',
            metavar=("FIELDNAME1='ICON1'", "FIELDNAME2='ICON2'"),
            dest='icon_pairs',
            help="A mapping of field names to icons.",
        )

        fields_group.add_argument(
            '-s', '--separator',
            type=ArgFormatter.ToTuple(length=2),
            metavar="'FIELD_SEPARATOR'",
            dest='separators',
            help=(
                "The character used for joining fields. "
                "Only valid with --field/-f options."
            ),
        )

        fields_group.add_argument(
            '--join-empty', '-j',
            action='store_true',
            dest='join_empty_fields',
            help=(
                "Include empty field contents instead of hiding them. "
                "Only valid with --field/-f options."
            ),
        )

        fields_group.add_argument(
            '--options', '-o',
            action='extend',
            nargs='+',
            metavar=('FIELD1.OPTION=VAL', 'FIELD2.OPTION=VAL'),
            dest='field_options',
            help="Arbitrarily set options for discrete Fields using dot-attribute syntax.",
        )

        self.add_argument(
            '-r', '--refresh',
            type=float,
            dest='refresh_rate',
            help=(
                "The bar's refresh rate in seconds per cycle."
            ),
        )

        self.add_argument(
            '--config', '-c',
            metavar='FILE',
            dest='config_file',
            help=(
                "The config file to use for default settings."
            ),
        )

        self.add_argument(
            '--count', '-n',
            type=int,
            metavar='TIMES',
            dest='count',
            help=(
                "Print the bar this many times, then exit."
            ),
        )

        self.add_argument(
            '--debug',
            action='store_true',
            help="Use debug mode. (Not implemented)",
        )

    def quit(self, message: str = "Exiting...") -> NoReturn:
        '''Print a message and exit the program.'''
        self.exit(1, message + '\n')


class HighlightMethod(Enum):
    '''Ways to present default option names.'''
    CAPITALIZE = 'CAPITALIZE'
    # UNDERLINE = 'UNDERLINE'
    # BOLD = 'BOLD'


class OptionsAsker:
    '''
    A tool for presenting options and gathering user input.

    :param opts: A mapping of options
    :type opts: :class:`OptSpec`

    :param default: The default option name
    :type default: :class:`str`

    :param question: Give the user context, defaults to ``""``
    :type question: :class:`str`

    :param case_sensitive: Options are case-sensitive, defaults to ``False``
    :type case_sensitive: :class:`bool`

    :param highlight_method: How to differentiate the default option
    :type highlight_method: :class:`HighlightMethod`

    :raises: :exc:`ValueError` if `default` is not a key in `opts`
    '''
    MISSING = None

    def __init__(
        self,
        opts: OptSpec,
        default: str,
        question: str = "",
        case_sensitive: bool = False,
        highlight_method: HighlightMethod | None = HighlightMethod.CAPITALIZE,
    ) -> None:
        # self.default_val = default_val
        if default not in opts:
            raise ValueError(
                f"The option dict {opts!r} "
                f"must contain the provided default option, {default!r}"
            )
        # self.opts = opts.copy()
        self.default = default
        self.default_val = opts[default]
        self.question = question
        self.case_sensitive = case_sensitive
        self.choices = opts.copy()

        self.optstrings = self.gen_optstrings(highlight_method)

        self.choices[''] = self.default_val

    def gen_optstrings(
        self,
        highlight_method: HighlightMethod | None = HighlightMethod.CAPITALIZE,
    ) -> tuple[OptName]:
        '''
        A tuple of option names with the default highlighted.

        :param highlight_method: How the default option should be differentiated,
            defaults to :obj:`HighlightMethod.CAPITALIZE`
        :type highlight_method: :class:`HighlightMethod`

        :returns: A tuple of option names
        :rtype: :class:`tuple[OptName]`
        '''
        match highlight_method:
            case HighlightMethod.CAPITALIZE:
                optstrings = (
                    choice.upper() if val == self.default_val # in default
                    else choice
                    for choice, val in self.choices.items()
                )
            case _:
                optstrings = self.choices

        return tuple(optstrings)

    def ask(
        self,
        prompt: str = None,
        repeat_prompt: bool = True,
    ) -> Any:
        '''
        Show a prompt and get user input for an option.
        Block until a valid option is given.

        :param prompt: Shows possible option names for the user to input
        :type prompt: :class:`string`, optional

        :param repeat_prompt: Repeat the whole prompt if the user enters an invalid option,
            defaults to ``True``
            Setting this to ``False`` is useful if :attr:`OptionsAsker.question` is very long.
        :type repeat_prompt: :class:`bool`

        :returns: The option value chosen by the user
        :rtype: :class:`Any`
        '''
        answer = self.MISSING
        options = f"[{'/'.join(self.optstrings)}]"
        if prompt is None:
            prompt = f"{self.question} {options} " if self.question else f"{options} "

        prompted = False
        while answer not in self.choices:
            answer = input(prompt)
            if not self.case_sensitive:
                answer = answer.casefold()
            if repeat_prompt or prompted:
                continue
            prompt = options + " "
        return self.choices.get(answer)

