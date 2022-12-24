import sys
from typing import NoReturn

from . import cli, Bar


def main() -> None | NoReturn:
    '''Run the command line utility.'''
    try:
        cfg = cli.get_config()
    except KeyboardInterrupt:
        print()
        sys.exit(1)
    bar = Bar.from_dict(cfg.bar_spec)
    bar.run()

main()

