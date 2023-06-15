import shlex
from io import StringIO
from os import PathLike


FILE = 'test_config.conf'


class Parser(shlex.shlex):
    '''
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        # Use newlines as tokens to find unset variables:
        self.whitespace = ' \t\r'

    def parse(self, file: PathLike):
        print(f"Parsing {file!r}:")
        with open(file, 'r') as f:
            self.text = f.read()
        self.instream = StringIO(self.text)

        return self.build_dict()
        self.prev_tok = None

    def parse_token(self):
        tok = self.get_token()

        # if tok != '\n':
            # print()
            # print(f"{self.prev_tok = !r}")
            # print(f"{tok = !r}")

        thing = None

        if tok == '':
            print("EOF")
            return tok

        elif tok == '=':
            target = self.prev_tok
            # print(f"Assigning to {target!r}")
            value = self.parse_token()
            if value in ('\n',):
                value = None
            print(f"{target!r} = {value!r}")
            thing = {target: value}
            tok = value

        # Maybe not necessary:
        elif tok in ('\n',):
            if self.prev_tok == '=':
                print("An empty assignment!")
                thing = None
            else:
                # Don't bother remembering newlines in `prev_tok`
                return tok
                # thing = self.parse_token()

        elif tok == '.':
            base = self.prev_tok
            attr_or_decimal = self.parse_token()
            if attr_or_decimal.isnumeric():
                if base.isnumeric():
                    # '1.234'
                    identifier = base + tok + attr_or_decimal
                else:
                    # '.234'
                    identifier = tok + attr_or_decimal
                
                thing = float(identifier)
                
            elif base.isidentifier() and attr_or_decimal.isidentifier():
                identifier = base + tok + attr_or_decimal
                # 'key.opt'
                thing = {identifier: {}}

            # print(thing)

            self.prev_tok = identifier
                

        elif tok == '{':
            print(f"Mapping to {self.prev_tok!r}")
            target = self.prev_tok

            assignments = {}
            while (maybe_assign := self.parse_token()) != '}':

                if maybe_assign in ('\n', ','):
                    continue
                if isinstance(maybe_assign, str):
                    # A symbol: the first half of an assignment:
                    continue
                if tuple(maybe_assign.values())[-1] == {}:
                    # Waiting for an assignment
                    continue

                assignment = '='.join(map(str, *maybe_assign.items()))
                print(f"Updating {target!r} with {assignment}")
                assignments.update(maybe_assign)

            # This causes issues becuase I need to update enclosing
            # dicts instead of making the target use dot-attribute syntax:
            thing = {target: assignments}
            print(f"Mapping complete: {thing}")

            self.prev_tok = '}'

        elif tok == '[':
            thing = []
            while (maybe_val := self.parse_token()) != ']':
                if maybe_val in ('\n', ','):
                    continue
                thing.append(maybe_val)

            self.prev_tok = ']'

        else:
            # An atom, either a symbol or a literal string:
            for speech_char in ('"', "'"):
                tok = tok.strip(speech_char)
            thing = tok
            self.prev_tok = tok

        return thing

    def build_dict(self):
        config = {}
        self.prev_tok = None
        while True:
            thing = self.parse_token()
            if thing == '':
                break

            if isinstance(thing, dict):
                config.update(thing)

        self.prev_tok = None
        return config


if __name__ == '__main__':
    print(Parser().parse(FILE))

