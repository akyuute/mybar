import sys

from .bar import Bar
from . import constants


def main() -> None:
    '''Run the command line utility.'''
    try:
        bar = Bar.from_cli()
        bar.run()
    except KeyboardInterrupt:
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()

