import re
import sys
from enum import Enum
from io import StringIO
from os import PathLike
from typing import NamedTuple, TypeAlias

LineNo: TypeAlias = int
TokenValue: TypeAlias = str
Line: TypeAlias = tuple[LineNo, list[TokenValue]]
Index: TypeAlias = int
# AST: TypeAlias = list




FILE = 'test_config.conf'

class TokenKind(Enum):
    ASSIGN = 'ASSIGN'
    DOT_LOOKUP = 'DOT_LOOKUP'
    LIT_FLOAT = 'LIT_FLOAT'
    LIT_INT = 'LIT_INT'
    LIT_STR = 'LIT_STR'
    SYMBOL = 'SYMBOL'
    SYNTAX = 'SYNTAX'

class Token(NamedTuple):
    value: TokenValue
    place: tuple[LineNo, Index]
    kind: TokenKind


class MyTokenizer:

    rules = (
        # Ignore
        (re.compile(r'^\#.*\n*'), None),  # Skip comments
        (re.compile(r'^\s+'), None),  # Skip whitespace

        (re.compile(r'^\n+'), 'NEWLINE'),  # To find empty assignments

        # Data types
        (re.compile(r'^\d*\.\d(\d|_)+'), 'FLOAT'),
        (re.compile(r'^\d+'), 'NUMBER'),
        (re.compile(
            r'^(?P<speech_char>["\'`]{3}|["\'`]{1})'
            r'((?!(?P=speech_char)).*)'
            r'(?P=speech_char)'
        ), 'STRING'),

        # Operators
        (re.compile(r'^='), 'ASSIGN'),
        (re.compile(r'^\.'), 'ATTRIBUTE'),

        # Syntax
        (re.compile(r'^\('), 'L_PAREN'),
        (re.compile(r'^\)'), 'R_PAREN'),
        (re.compile(r'^\['), 'L_BRACKET'),
        (re.compile(r'^\]'), 'R_BRACKET'),
        (re.compile(r'^\{'), 'L_CURLY_BRACE'),
        (re.compile(r'^\}'), 'R_CURLY_BRACE'),
        (re.compile(r'^[a-zA-Z_]+'), 'SYMBOL'),
    )

    def __init__(
        self,
        string: str = None,
        file: PathLike = None,
    ) -> None: 
        if string is None:
            if file is None:
                file = FILE
                # raise ValueError("Missing source")

            with open(file, 'r') as f:
                string = f.read()

        self._string = string
        self._file = file
        self._stream = StringIO(self._string)
        self._lookahead = None
        self._cursor = 0
        self.eof = ''

    def error_leader(self) -> str:
        file = self._file if self._file is not None else ''
        return f"File {file!r}, line {self.lineno()}: "

    def lineno(self) -> int:
        return self._string[:self._cursor].count('\n') + 1
    
    def at_eof(self) -> bool:
        return self._cursor == len(self._string)

    def lex(self, string: str = None) -> list[Token]:
        if string is not None:
            self._string = string

        tokens = []
        while True:
            tok = self.get_token()
            if tok == '':
                break
            print(tok)
            tokens.append(tok)
        return tokens

    def get_token(self) -> Token:
        if self.at_eof():
            return self.eof

        s = self._string[self._cursor:]
        for test, kind in self.rules:
            m = re.match(test, s)

            if m is None:
                # Rule not matched
                continue

            if kind is None:
                # Skip whitespace
                self._cursor += len(m.group())
                return self.get_token()

            tok = Token(value=m.group(), place=self._cursor, kind=kind)
            self._cursor += len(m.group())
            return tok

        raise SyntaxError(
            self.error_leader()
            + f"Unexpected token: {s!r}"
        ).with_traceback(None)


class Parser:
    pass

class Tree:
    pass

if __name__ == '__main__':
    # p = Parser()
    # t = MyTokenizer(string=' '.join(sys.argv[1:]))
    t = MyTokenizer()
    tokens = t.lex()
    # print(tokens)
    # return tokens
