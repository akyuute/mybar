from mybar.base import Bar, Config
from mybar import cli 

from typing import NoReturn


def main() -> NoReturn:
    '''Run the command line utility.'''
    cfg = cli.gather_config()
    bar = cfg.to_bar()
    bar.run()

main()

