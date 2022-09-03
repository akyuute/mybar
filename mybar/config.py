import json
import os
from utils import make_error_message
# from .mybar import Bar


class Config:
    def __init__(self, file: os.PathLike = None): # Path = None):
        # Get the config file name if passed as a command line argument
        cli_parser = ArgumentParser()
        cli_parser.add_argument('--config', nargs='+')
        config_file = cli_parser.parse_args(sys.argv[1:]).config or file

        self.file = config_file
        with open(self.file, 'r') as f:
            self.data = json.load(f)
            self.raw = f.read()

    def write_file(self, file: os.PathLike = None, obj: dict = None):
        if file is None:
            file = self.file
        if obj is None:
            obj = self.data
        # return json.dumps(obj)
        with open(file, 'w') as f:
            json.dump(obj, f)


    def get_bar(self) -> Bar:
        return Bar.from_dict(self.data)

