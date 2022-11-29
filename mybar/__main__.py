from mybar.base import Bar, Config
from mybar import cli 

from typing import NoReturn


def main() -> NoReturn:
    '''Run the command line utility.'''
    # cfg = cli.gather_config()

    parser = cli.Parser()
    try:
        options = parser.parse_args()
    except cli.UnrecoverableError as e:
        parser.error(e.msg)
    
    try:
        cfg = Config(opts=options)
    except OSError as e:
        errmsg = f"{parser.prog}: error: {e}"
        parser.exit(1, message=errmsg + '\n')

    bar = cfg.to_bar()
    bar.run()

main()

