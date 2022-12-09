import sys
from typing import NoReturn

from mybar import cli 


def main() -> None | NoReturn:
    '''Run the command line utility.'''
    try:
        cfg = cli.make_initial_config()
    except KeyboardInterrupt:
        print()
        sys.exit(1)
    bar = cfg.to_bar()
    bar.run()

main()

