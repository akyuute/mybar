__all__ = (
    'Parser',
    'OptionsAsker',
)


import os.path
from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
from enum import Enum
from os import PathLike

from . import __version__
from .constants import CONFIG_FILE
from .errors import AskWriteNewFile, CLIFatalError, CLIUsageError
from .namespaces import BarConfigSpec, CmdOptionSpec, FieldSpec
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

    def parse_args(
        self,
        args: list[str] = None
    ) -> tuple[BarConfigSpec, CmdOptionSpec]:
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

        # For options that control specific fields:
        if 'field_options' in params:
            fields = self.process_field_options(
                params.pop('field_options')
            )
            params['field_definitions'] = fields

        # For options that control what the command does:
        keys = (
            CmdOptionSpec.__optional_keys__ ^ CmdOptionSpec.__required_keys__
        )
        cmd_options = {k: params.pop(k) for k in keys if k in params}
        return params, cmd_options

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
        fields_or_tmpl = self.add_mutually_exclusive_group()
        fields_or_tmpl.add_argument(
            '--template', '-t',
            dest='template',
            help=(
                "A curly-brace-delimited format string."
                " Not valid with --fields/-f options."
            ),
            metavar="'TEMPLATE'",
        )

        fields_or_tmpl.add_argument(
            '--fields', '-f',
            action='extend',
            dest='field_order',
            help=(
                "A list of fields to be displayed."
                " Not valid with --template/-t options."
            ),
            metavar=('FIELDNAME1', 'FIELDNAME2'),
            nargs='+',
            # type=ArgFormatter.SplitFirst(','),
        )

        fields_group = self.add_argument_group(
            title="Options for fields",
            description="These options are not valid when using --template/-t."
        )

        fields_group.add_argument(
            '--icons',
            action='extend',
            dest='icon_pairs',
            help="A mapping of field names to icons.",
            metavar=("FIELDNAME1='ICON1'", "FIELDNAME2='ICON2'"),
            nargs='+',
        )

        fields_group.add_argument(
            '--separator', '-s',
            dest='separators',
            help=(
                "The character used for joining fields."
                " Only valid with --field/-f options."
            ),
            metavar="'FIELD_SEPARATOR'",
            type=ArgFormatter.ToTuple(length=2),
        )

        fields_group.add_argument(
            '--join-empty', '-j',
            action='store_true',
            dest='join_empty_fields',
            help=(
                "Include empty field contents instead of hiding them."
                " Only valid with --field/-f options."
            ),
        )

        fields_group.add_argument(
            '--options', '-o',
            action='extend',
            dest='field_options',
            help=(
                "Set arbitrary options for discrete Fields using"
                " dot-attribute syntax."
            ),
            metavar=("'FIELD1.OPTION=VAL'", "'FIELD2.OPTION=VAL'"),
            nargs='+',
        )

        self.add_argument(
            '-r', '--refresh',
            dest='refresh_rate',
            help=(
                "The bar's refresh rate in seconds per cycle."
            ),
            type=float,
        )

        self.add_argument(
            '--count', '-n',
            dest='count',
            help=(
                "Print the bar this many times, then exit."
            ),
            metavar='TIMES',
            type=int,
        )

        self.add_argument(
            '--config', '-c',
            dest='config_file',
            help=(
                "The config file to use for default settings."
            ),
            metavar='FILE',
        )

        self.add_argument(
            '--dump', '-d',
            action='store_const',
            const=4,
            dest='dump_config',
            help=(
                "Instead of running the Bar, print a JSON config using"
                " options specified in the command."
                " Optionally pass a number of spaces by which to indent."
            ),
        )

        self.add_argument(
            '--dump-raw', '-D',
            action='store_const',
            const=None,
            dest='dump_config',
            help=(
                "Instead of running the Bar, print a JSON config using"
                " options specified in the command."
            ),
        )

        self.add_argument(
            '--debug',
            action='store_true',
            help="Use debug mode. (Not implemented)",
        )

        self.add_argument(
            '--version', '-v',
            action='version',
            version=f"{__package__} {__version__}",
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


def welcome_new_users() -> None:
    msg = (
        f"-- {PROG} --"
    )
    print(msg)


##class ConfigWizard:
class FileManager:

    @classmethod
    def _get_write_new_approval(
        cls,
        file: PathLike,
        dft_choice: bool
    ) -> bool:
        '''
        Get approval to write a new config file using
            :class:`cli.OptionsAsker`.

        :param file: The path to the config file
        :type file: :class:`PathLike`

        :param dft_choice: The default option to present to the user
            (``False`` means do not write)
        :type dft_choice: :class:`bool`
        '''
        question = f"Would you like to write a new config file at {file!r}?"

        if not os.path.exists(file):
            maybe_default = ' '
            if file == CONFIG_FILE:
                welcome_new_users()
                cls._maybe_make_config_dir()
                maybe_default = ' default '
            msg = (
                f"The{maybe_default}config file at {file!r} does not exist."
            )
            print(msg)
            question = "Would you like to make it now?"

        write_options = {'y': True, 'n': False}
        default = 'ny'[dft_choice]
        handler = OptionsAsker(write_options, default, question)

        write_new_file_ok = handler.ask()
        return write_new_file_ok

    @staticmethod
    def _maybe_make_config_dir() -> None:
        '''
        Make a '.config' directory if nonexistent.
        '''
        directory = os.path.dirname(CONFIG_FILE)
        if not os.path.exists(directory):
            os.mkdir(directory)

