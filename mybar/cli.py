from argparse import ArgumentParser, SUPPRESS, Namespace, HelpFormatter
import re
import sys

from mybar.base import AskWritingToRequestedFile, Bar, Config


from typing import NoReturn, TypeAlias
ConfigSpec: TypeAlias = dict


PROG = __package__

class MyFormatter(HelpFormatter):
    def _format_actions_usage(self, actions, groups):
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            if not group._group_actions:
                raise ValueError(f'empty group {group}')

            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                end = start + len(group._group_actions)
                if actions[start:end] == group._group_actions:
                    for action in group._group_actions:
                        group_actions.add(action)
                    if not group.required:
                        if start in inserts:
                            inserts[start] += ' ['
                        else:
                            inserts[start] = '['
                        if end in inserts:
                            inserts[end] += ']'
                        else:
                            inserts[end] = ']'
                    else:
                        if start in inserts:
                            inserts[start] += ' ('
                        else:
                            inserts[start] = '('
                        if end in inserts:
                            inserts[end] += ')'
                        else:
                            inserts[end] = ')'
                    for i in range(start + 1, end):
                        inserts[i] = '|'

        # collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # suppressed arguments are marked with None
            # remove | separators for suppressed arguments
            if action.help is SUPPRESS:
                parts.append(None)
                if inserts.get(i) == '|':
                    inserts.pop(i)
                elif inserts.get(i + 1) == '|':
                    inserts.pop(i + 1)

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]

                # add the action string to the list
                parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
################option_string = action.option_strings[0]
                option_string = '|'.join(action.option_strings)

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = action.format_usage()

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = '%s %s' % (option_string, args_string)

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = '[%s]' % part

                # add the action string to the list
                parts.append(part)

        # insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # join all the action items with spaces
        text = ' '.join([item for item in parts if item is not None])

        # clean up separators for mutually exclusive groups
        open = r'[\[(]'
        close = r'[\])]'
        text = re.sub(r'(%s) ' % open, r'\1', text)
        text = re.sub(r' (%s)' % close, r'\1', text)
        text = re.sub(r'%s *%s' % (open, close), r'', text)
        text = re.sub(r'\(([^|]*)\)', r'\1', text)
        text = text.strip()

        # return the text
        return text



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
            prog=PROG,
            argument_default=SUPPRESS,
            # formatter_class=MyFormatter,
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
            '-s', '--separator',
            metavar="'FIELD_SEPARATOR'",
            dest='separator',
            help="",
        )

        self.add_argument(
            '-r', '--refresh',
            type=float,
            dest='refresh_rate',
            help="",
        )

        self.add_argument(
            '--config', '-c',
            metavar='FILE',
            dest='config_file',
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
            '--debug',
            action='store_true',
            help="",
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

