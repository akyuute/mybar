import os.path
import sys
from typing import NoReturn

from .bar import Bar
from . import constants


def main() -> NoReturn | None:
    '''Run the command line utility.'''
    first_use = (not os.path.exists(constants.CONFIG_FILE))
    if first_use:
        from .cli import OptionsAsker

        icon_examples = ' '.join(constants.FONTAWESOME_ICONS)
        question = (
            f"If FontAwesome is installed on your system, \n"
            f"would you like to use FontAwesome Field icons "
            f"( {icon_examples} ) \n"
            f"instead of default ASCII icons?"
        )
        options = {'n': False, 'y': True}
        default = 'n'
        asker = OptionsAsker(options, default, question)

        use_fontawesome = default
        try:
            use_fontawesome = asker.ask(repeat_prompt=False)
        except KeyboardInterrupt:
            print()
            sys.exit(1)

        choice = "FontAwesome icons" if use_fontawesome else "default icons"
        print(f"Using {choice}...")
        constants.USING_FONTAWESOME = use_fontawesome

    try:
        bar = Bar.from_cli()
    except KeyboardInterrupt:
        print()
        sys.exit(1)
    bar.run()


if __name__ == '__main__':
    main()

