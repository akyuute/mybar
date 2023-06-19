import shlex
from io import StringIO
from os import PathLike
from queue import Queue


FILE = 'test_config.conf'


class Parser(shlex.shlex):
    __NESTED = 'NESTED'

    '''
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        # Use newlines as tokens to find unset variables:
        self.whitespace = self.whitespace.replace('\n', '')

    def parse(self, file: PathLike):
        with open(file, 'r') as f:
            self.text = f.read()
        self.instream = StringIO(self.text)
        self.infile = file

        # print(f"Parsing {file!r}:")
        return self.build_dict()
        self.prev_tok = None

    def build_dict(self):
        config = {}
        self.prev_tok = None
        while True:
            parsed = self.parse_token()
            if parsed == '':
                break

            if isinstance(parsed, dict):
                config.update(parsed)

        self.prev_tok = None
        return config

    def parse_token(self):
        tok = self.get_token()
        parsed = None

        if tok == '':
            # EOF
            return tok

        elif tok in ('\n',):
            # Don't bother remembering newlines in `prev_tok`:
            return tok

        elif tok == '=':
            target = self.prev_tok
            self.prev_tok = tok  # Tell '{' who called it
            value = self.parse_token()
            if value in ('\n', ','):
                value = None
            return {target: value}

        elif tok == '.':
            base = self.prev_tok
            attr_or_decimal = self.parse_token()
            if attr_or_decimal.isnumeric():
                if base.isnumeric():
                    # '1.234'
                    num = base + tok + attr_or_decimal
                else:
                    # '.234'
                    num = tok + attr_or_decimal
                
                parsed = float(num)

            elif base.isidentifier() and attr_or_decimal.isidentifier():
                # 'foo.bar'
                parsed = {self.__NESTED: True, base: attr_or_decimal}

            return parsed

        elif tok == '{':
            target = self.prev_tok
            assignments = {}
            while (maybe_assign := self.parse_token()) != '}':

                if maybe_assign in ('\n', ','):
                    continue

                if isinstance(maybe_assign, str):
                    # '{foo ' ... = bar
                    # A symbol, the first half of an assignment:
                    continue

                if not isinstance(maybe_assign, dict):
                    msg = f"{self.error_leader()} A weird thing happened"
                    raise ValueError(msg)

                if maybe_assign.pop(self.__NESTED, False):
                    # 'kwargs.foo'
                    base, attr = maybe_assign.popitem()

                    value = self.parse_token()
                    # while (value := self.parse_token()).pop(self.__NESTED, False):
                    # 'kwargs.foo = bar'
                    if not isinstance(value, dict):
                        msg = self.error_leader()
                        msg += "An interesting thing happened"
                        raise ValueError(msg)

                    if base not in assignments:
                        assignments[base] = value
                    else:
                        assignments[base].update(value)

                else:
                    assignments.update(maybe_assign)

            self.prev_tok = '}'

            # Avoid recursively assigning to `target`:
            if target.isidentifier():
                return {target: assignments}

            return assignments

        elif tok == '[':
            array = []
            while (elem := self.parse_token()) != ']':
                if elem in ('\n', ','):
                    continue
                array.append(elem)

            self.prev_tok = ']'
            return array

        else:
            # An atom, either a symbol or a literal string:
            for speech_char in ('"', "'"):
                if tok.startswith(speech_char):
                    if not tok.endswith(speech_char):
                        msg = f"{self.error_leader()} Missing a {speech_char}"
                        raise ValueError(msg)
                    tok = tok.strip(speech_char)
                    break
            self.prev_tok = tok
            return tok


if __name__ == '__main__':
    print(Parser().parse(FILE))

