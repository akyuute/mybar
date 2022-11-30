from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
import re
import sys

from mybar.base import AskWritingToRequestedFile, Bar, Config
##from mybar.utils import join_options


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
##        self.conflicting_args: list[set] = []
##        self.conflicting_args.append({'fmt', 'field_order'})

    def parse_args(self, args: None | list[str] = None) -> ConfigSpec:
        '''Parse command line arguments and return a dict of options.'''
        # Use vars() because dict items are more portable than attrs.
        opts = vars(super().parse_args(args))
        # self.check_conflicting_args(opts)
        params = self.process_complex_args(opts)
        return params

##    def check_conflicting_args(self, opts: dict) -> None | NoReturn:
##
##        # if 'fmt' in opts and 'field_order' in opts:
##        for coll in self.conflicting_args:
##            if len(matches := coll.intersection(opts)) >= 2:
##                args = [
##                    '/'.join(arg.option_strings) for arg in self._actions
##                    if arg.dest in matches
##                ]
##                together = join_options(
##                    args,
##                    sep=', ' if len(matches) > 2 else ' ',
##                    final_sep='and '
##                )
##                err = f"Options {together} are mutually exclusive."
##                # raise Exception(err)
##                raise UsageError(err)

    def process_complex_args(self, opts: ConfigSpec) -> ConfigSpec:
        if (icons := opts.pop('icon_pairs', None)):
            opts['field_icons'] = dict(
                pair for item in icons
                if len(pair := item.split('=', 1)) == 2
            )
        return opts

    def split_list_args(args: list) -> list|None: ...

    def add_arguments(self) -> None:

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

