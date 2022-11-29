from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
import re
import sys

from mybar.base import AskWritingToRequestedFile, Bar, Config


from typing import NoReturn, TypeAlias
ConfigSpec: TypeAlias = dict


PROG = __package__


class UnrecoverableError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

class UsageError(UnrecoverableError):
    pass


class Parser(ArgumentParser):
    def __init__(self) -> None:
        super().__init__(
            prog=PROG,
            argument_default=SUPPRESS,
        )
        self.add_arguments()

    def parse_args(self, args: None | list[str] = None) -> ConfigSpec:
        opts = vars(super().parse_args(args))
        self.check_conflicting_options(opts)
        params = self.process_complex_args(opts)
        return params

    def check_conflicting_options(self, opts: dict) -> None | NoReturn:
        if 'fmt' in opts and 'field_order' in opts:
            err = (
                "--fields and --format options are mutually exclusive. "
                "(usage msg here)"
            )
            raise UsageError(err)

    def process_complex_args(self, opts: ConfigSpec) -> ConfigSpec:
        if (icons := opts.pop('icon_pairs', None)):
            opts['field_icons'] = dict(
                pair for item in icons
                if len(pair := item.split('=', 1)) == 2
            )
        return opts

    def split_list_args(args: list) -> list|None: ...

    def add_arguments(self) -> None:
        self.add_argument(
            '-f', '--fields',
            action='extend',
            nargs='+',
            # metavar='FIELD1[,FIELD2, ...]',
            metavar=('FIELDNAME1', 'FIELDNAME2'),
            dest='field_order',
            # type=(lambda f: f.split(',', 1)),
            help="A list of fields to be displayed. Not valid with --format/-m options.",
        )

        self.add_argument(
            '--icons',
            action='extend',
            nargs='+',
            metavar=("FIELDNAME1='ICON1'", "FIELDNAME2='ICON2'"),
            dest='icon_pairs',
            help="A mapping of field names to icons.",
        )

        self.add_argument(
            '-m', '--format',
            metavar="'FORMAT_STRING'",
            dest='fmt',
            help=(
                "A curly-brace-delimited format string. "
                "Not valid with --fields/-f options."
            ),
        )

        self.add_argument(
            '-s', '--separator',
            metavar="'FIELD_SEPARATOR'",
            dest='separator',
            help=(
                "The character used for joining fields. "
                "Only valid with --field/-f options."
            ),
        )

        self.add_argument(
            '-r', '--refresh',
            type=float,
            dest='refresh_rate',
            help=(
                "The bar's refresh rate in cycles/second."
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
            '--join-empty', '-j',
            action='store_true',
            dest='join_empty_fields',
            help=(
                "Include empty field contents instead of hiding them. "
                "Only valid with --field/-f options."
            ),
        )

        self.add_argument(
            '--once', '-o',
            action='store_true',
            dest='run_once',
            help=(
                "Only run the bar once rather than continuously."
            ),
        )

        self.add_argument(
            '--debug',
            action='store_true',
            help="Use debug mode.",
        )


def gather_config():
    parser = Parser()
    try:
        options = parser.parse_args()
    except UnrecoverableError as e:
        parser.error(e.msg)
    
    try:
        cfg = Config(opts=options)

    except AskWritingToRequestedFile as e:
        file = e.requested_file
        errmsg = (
            f"{parser.prog}: error: \n"
            f"The config file at {file} does not exist."
        )
        prompt = f"Would you like to make it? [y/N] "

        choices = ('n', 'y', '')
        answer = None
        print(errmsg)
        while answer not in choices:
            answer = input(prompt)
            match answer.lower():
                case '' | 'n':
                    parser.exit(1, message="Exiting...\n")
                case 'y':
                    Config.write_file(file, options)
                    cfg = Config(opts=options)

    except OSError as e:
        errmsg = f"{parser.prog}: error: {e}"
        parser.exit(1, message=errmsg + '\n')

    return cfg

