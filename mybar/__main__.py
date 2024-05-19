import os.path
import sys

from .bar import Bar, BarConfig
from .cli import ArgParser, CLIUsageError
from .constants import CONFIG_FILE


def main() -> None:
    '''Run the command line utility.'''
    try:
        parser = ArgParser()
        try:
            bar_options, command_options = parser.parse_args()
        except CLIUsageError as e:
            parser.error(e.msg)  # Shows usage

        file = command_options.pop('config_file', CONFIG_FILE)
        absolute = os.path.abspath(os.path.expanduser(file))

        try:
            config = BarConfig.from_file(absolute, overrides=bar_options)

        # Handle missing config files:
        except FileNotFoundError as e:
            if command_options:
                if 'dump_config' in command_options:
                    # Skip writing:
                    pass

            write_ok = BarConfig.write_with_approval(
                absolute, overrides=bar_options
            )
            if write_ok:
                config = BarConfig.from_file(file, overrides=bar_options)
            else:
                # Forget all this config file business.
                # Our new user is in a hurry, so
                # just give them what they need:
                config = BarConfig(bar_options)

        # Permissions error:
        except OSError as e:
            e.add_note("Exiting...")
            raise e from None

        except KeyboardInterrupt:
            parser.quit()

        # Handle options that alter the behavior of the command itself:
        if command_options:
            if 'dump_config' in command_options:
                indent = command_options.pop('dump_config', None)
                parser.quit(BarConfig._as_json(config, indent=indent))

        bar = Bar.from_config(config)
        bar.run()

    except KeyboardInterrupt:
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()

