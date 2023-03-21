__all__ = (
    'Parser',
    'OptionsAsker',
)


from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
from enum import Enum

from .errors import AskWriteNewFile, FatalError, CLIUsageError
from ._types import BarTemplateSpec, FieldName, OptName, OptSpec

from typing import Any, Callable, NoReturn, TypeAlias


### Constants ###
PROG = __package__

def SplitFirst(char: str) -> Callable[[str], str]:
    return (lambda f: f.split(char, 1))

def ToTuple(length: int) -> Callable[[Any], tuple]:
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

    def parse_args(self, args: None | list[str] = None) -> BarTemplateSpec:
        '''Parse command line arguments and return a dict of options.'''
        # Use vars() because dict items are more portable than attrs.
        opts = vars(super().parse_args(args))
        params = self.process_assignment_args(opts)
        return params

    def process_assignment_args(self,
        opts: BarTemplateSpec,
        assignments: dict[FieldName, str] = None
    ) -> BarTemplateSpec:
        '''Make dicts from key-value pairs in assignment args.'''
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

    def add_arguments(self) -> None:
        '''Equip the parser with all its arguments.'''
        fields_or_fmt = self.add_mutually_exclusive_group()
        fields_or_fmt.add_argument(
            '-m', '--format',
            metavar="'FORMAT_STRING'",
            dest='fmt',
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
            # type=SplitFirst(','),
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
            type=ToTuple(length=2),
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
            '--once', '-o',
            action='store_true',
            dest='run_once',
            help=(
                "Run the bar once rather than continuously."
            ),
        )

        self.add_argument(
            '--debug',
            action='store_true',
            help="Use debug mode.",
        )

    def quit(self, message: str = "Exiting...") -> NoReturn:
        self.exit(1, message + '\n')


class HighlightMethod(Enum):
    CAPITALIZE = 'CAPITALIZE'
    # UNDERLINE = 'UNDERLINE'
    # BOLD = 'BOLD'


class OptionsAsker:
    MISSING = None

    def __init__(self,
        opts: OptSpec,
        default: str | tuple[str],
        question: str = "Foo?",
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

    def gen_optstrings(self,
        highlight_method: HighlightMethod | None = HighlightMethod.CAPITALIZE,
    ) -> tuple[OptName]:

        ### Only needed if matching OptName, not default_val:
        default = self.default
        if isinstance(default, str):
            default = (default,)
        elif hasattr(default, '__iter__'):
            pass
        else: raise TypeError()
        ###

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

    def ask(self,
        prompt: str = None,
        highlight_method: HighlightMethod | None = HighlightMethod.CAPITALIZE,
        repeat_prompt: bool = True,
    ) -> Any:
        answer = self.MISSING
        options = f"[{'/'.join(self.optstrings)}]"
        if prompt is None:
            prompt = f"{self.question} {options} "

        prompted = False
        while answer not in self.choices:
            answer = input(prompt)
            if not self.case_sensitive:
                answer = answer.casefold()
            if repeat_prompt or prompted:
                continue
            prompt = options + " "
        return self.choices.get(answer)

