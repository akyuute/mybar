from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
import re
import sys

from mybar.base import Bar, Config
from mybar.errors import AskWriteNewFile


from typing import Any, NoReturn, TypeAlias
ConfigSpec: TypeAlias = dict
OptName: TypeAlias = str
CurlyBraceFormatString: TypeAlias = str


PROG = __package__


class UnrecoverableError(Exception):
    '''Base class for errors that cause the program to exit.'''
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

class UsageError(UnrecoverableError):
    '''Raised when the command is used incorrectly.'''
    pass


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

    def parse_args(self, args: None | list[str] = None) -> ConfigSpec:
        '''Parse command line arguments and return a dict of options.'''
        # Use vars() because dict items are more portable than attrs.
        opts = vars(super().parse_args(args))
        params = self.process_assignment_args(opts)
        return params

    def process_assignment_args(self,
        opts: ConfigSpec,
        assignments: dict[str, str] = None
    ) -> ConfigSpec:
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
            # type=(lambda f: f.split(',', 1)),
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
            metavar="'FIELD_SEPARATOR'",
            dest='separator',
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
                "The bar's refresh rate in cycles per second."
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


class DialogHandler:
    UNASSIGNED = None

    def __init__(self,
        opts: dict[Any, OptName],
        default_val: Any,
        prompt: CurlyBraceFormatString = "Foo? [{}] ",
        display_default: bool = True,
        case_sensitive: bool = False
    ) -> None:
        self.opts = opts
        self.default_val = default_val
        self.prompt = prompt
        self.display_default = display_default
        self.case_sensitive = case_sensitive

        optstrings = opts.copy()
        default_str = optstrings[default_val]
        optstrings[default_val] = default_str.upper()
        self.optstrings = tuple(optstrings.values())

        self.choices = {
            v if case_sensitive
            else v.lower(): k for k, v in opts.items()
        }
        self.choices[''] = default_val

    def handle_dialog(self) -> dict[str]:
        answer = self.UNASSIGNED
        prompt = self.prompt.format('/'.join(self.optstrings))
        while answer not in self.choices:
            answer = input(prompt)
            if not self.case_sensitive:
                answer = answer.lower()
        return self.choices.get(answer)

def make_initial_config(write_new_file_dft: bool = False) -> Config:
    '''Parse args from stdin and return a new Config.'''
    parser = Parser()
    try:
        bar_options = parser.parse_args()
        cfg = Config(opts=bar_options)

    except AskWriteNewFile as e:
        file = e.requested_file
        errmsg = (
            f"{parser.prog}: error: \n"
            f"The config file at {file} does not exist."
        )

        file_options = {True: 'y', False: 'n'}
        prompt = "Would you like to make it now? [{}] "
        handler = DialogHandler(file_options, write_new_file_dft, prompt)

        print(errmsg)
        write_new_file = handler.handle_dialog()
        if write_new_file:
            Config.write_file(file, bar_options)
            print(f"Wrote new config file at {file}")
            cfg = Config(opts=bar_options)
        else:
            parser.quit()

    except UnrecoverableError as e:
        parser.error(e.msg)

    except OSError as e:
        err = f"{parser.prog}: error: {e}"
        parser.quit(err)

    return cfg

