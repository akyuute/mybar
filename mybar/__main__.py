import os.path
import sys
from typing import NoReturn

from .bar import Bar
from . import constants


def main() -> NoReturn | None:
    '''Run the command line utility.'''
    try:
        bar = Bar.from_cli()
        bar.run()
    except KeyboardInterrupt:
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()

