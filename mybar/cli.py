from argparse import ArgumentParser, SUPPRESS, Namespace
import sys

from mybar.base import Config, Bar


from typing import NoReturn, TypeAlias
ConfigSpec: TypeAlias = dict


class UnrecoverableError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

class UsageError(UnrecoverableError):
    pass


class Parser(ArgumentParser):
    #TODO: Override print_usage!
    def __init__(self) -> None:
        super().__init__(
            prog=__package__,
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
        if (pairs := opts.pop('icon_pairs', None)):
            opts['field_icons'] = dict(
                pair.split('=', 1) for pair in pairs
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
            help="",
        )

        self.add_argument(
            '--icons',
            action='extend',
            nargs='+',
            metavar=("FIELDNAME1='ICON1'", "FIELDNAME2='ICON2'"),
            dest='icon_pairs',
            help="",
        )

        self.add_argument(
            '-m', '--format',
            metavar="'FORMAT_STRING'",
            dest='fmt',
            help="",
        )

        self.add_argument(
            '-s', '--sep', '--separator',
            metavar="'FIELD_SEPARATOR'",
            dest='separator',
            help="",
        )

        self.add_argument(
            '-r', '--ref', '--refresh',
            type=float,
            dest='refresh_rate',
            help="",
        )

        self.add_argument(
            '--join-empty', '-j',
            action='store_true',
            dest='join_empty_fields',
            help="",
        )

        self.add_argument(
            '--once', '-o',
            action='store_true',
            dest='run_once',
            help="",
        )

        self.add_argument(
            '--config', '-c',
            metavar='FILE',
            dest='config_file',
            help="",
        )

        self.add_argument(
            '--debug',
            action='store_true',
            help="",
        )

# argnames = [(a.option_strings[0], a.metavar or a.dest.upper()) for a in parser._actions]

