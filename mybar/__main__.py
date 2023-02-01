import sys
from typing import NoReturn

from .bar import Bar


def main() -> NoReturn | None:
    '''Run the command line utility.'''
    try:
        bar = Bar.from_cli()
    except KeyboardInterrupt:
        print()
        sys.exit(1)
    bar.run()


if __name__ == '__main__':
    main()

