'''
mybar
[--config FILE]
[--separator SEPARATOR] 
[--refresh REFRESH]
[...]
( [--fmt 'FORMAT'] | [--fields ...] )
'''

from argparse import ArgumentParser, SUPPRESS

parser = ArgumentParser(
    argument_default=SUPPRESS,
)

fields_or_format = parser.add_mutually_exclusive_group()

##############################
fields_or_format.add_argument(
    '-m', '--format',
    metavar="'FORMAT_STRING'",
    dest='fmt',
)

fields_or_format.add_argument(
    '-f', '--fields',
    action='extend',
    # nargs='+',
    metavar='FIELD1[,FIELD2, ...]',
    # type=(lambda f: f.split(',', 1)),
)
##############################

parser.add_argument(
    '-s', '--sep', '--separator',
    # default='|',
    metavar="'FIELD_SEPARATOR'",
    dest='separator',
)

parser.add_argument(
    '--refresh', '-r', '--ref',
    type=float,
    dest='refresh_rate',
)

parser.add_argument(
    '--join-empty', '-j',
    action='store_true',
)

parser.add_argument(
    '--once', '-o',
    action='store_true',
)

parser.add_argument(
    '--icons', '-i',
    action='extend',
    nargs='+',
    type=lambda s: s.split('=', 1),
    metavar=('FIELDNAME1=ICON1', 'FIELDNAME2=ICON2'),
)

parser.add_argument(
    '--config', '-c',
    # metavar='CONFIG_FILE',
    dest='config_file',
)

parser.add_argument(
    '--debug',
    action='store_true',
)

ns = parser.parse_args()
print(ns)

argnames = [(a.option_strings[0], a.metavar or a.dest.upper()) for a in parser._actions]


##    name: str = None,
##    func: Callable[..., str] = None,
##    icon: str = '',
##    fmt: str = None,
##    interval: float = 1.0,
##    align_to_seconds: bool = False,
##    overrides_refresh: bool = False,
##    threaded: bool = False,
##    constant_output: str = None,
##    run_once: bool = False,
##    bar=None,
##    args = None,
##    kwargs = None,
##    setup: Callable[..., Kwargs] = None,

##class Config:
##    def __init__(self, file: os.PathLike = None) -> None:
##        # Get the config file name if passed as a command line argument
##        cli_parser = ArgumentParser()
##        cli_parser.add_argument('--config')
##        config_file = (
##            cli_parser.parse_args(sys.argv[1:]).config
##            or file
##            or CONFIG_FILE
##        )
##
##        absolute = os.path.expanduser(config_file)
##        if not os.path.exists(absolute):
##            self.write_file(absolute)
##        self.file = absolute
##
##        self.data, self.text = self.read_file(absolute)
##
##    def __repr__(self) -> str:
##        cls = self.__class__.__name__
##        file = self.file
##        return f"{cls}({file=})"
##
##    def make_bar(self) -> Bar:
##        return Bar.from_dict(self.data)
##
##    def read_file(self, file: os.PathLike = None) -> tuple[BarSpec, str]:
##        if file is None:
##            file = self.file
##        with open(self.file, 'r') as f:
##            data = json.load(f)
##            text = f.read()
##        return data, text
##
##    def write_file(self,
##        file: os.PathLike = None,
##        obj: BarSpec = None
##    ) -> None:
##        if file is None:
##            file = self.file
##
##        # return json.dumps(obj)
##        obj = Bar._default_params.copy() if obj is None else obj
##
##        dft_bar = obj
##        dft_fields = Field._default_fields.copy()
##
##        for name, field in dft_fields.items():
##            new = dft_fields[name] = field.copy()
##            for param in ('name', 'func', 'setup'):
##                try:
##                    del new[param]
##                except KeyError:
##                    pass
##
##        dft_bar['field_definitions'] = dft_fields
##        dft_bar['field_order'] = Bar._default_field_order
##        self.defaults = dft_bar
##
##        # return self.defaults
##        with open(os.path.expanduser(file), 'w') as f:
##            json.dump(self.defaults, f, indent=4, ) #separators=(',\n', ': '))

