import re
import sys

from ast import AST, Assign, Name, Expr, Attribute, BinOp, UnaryOp, List, Constant
from ast import Set 
from enum import Enum
from io import StringIO
from itertools import chain
from os import PathLike
from typing import NamedTuple, NoReturn, TypeAlias


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]
Index: TypeAlias = int
Tree: TypeAlias = list
Tree: TypeAlias = Tree[Tree]


FILE = 'test_config.conf'


class Literal(Enum):
    LIT_FLOAT = 'LIT_FLOAT'
    LIT_INT = 'LIT_INT'
    LIT_STR = 'LIT_STR'


class Ignore(Enum):
    COMMENT = 'COMMENT'
    NEWLINE = 'NEWLINE'
    SPACE = 'SPACE'


class Symbol(Enum):
    IDENTIFIER = 'IDENTIFIER'
    KEYWORD = 'KEYWORD'

KEYWORDS = ()


class BinOp(Enum):
    ADD = '+'
    SUBTRACT = '-'
    ATTRIBUTE = '.'
    ASSIGN = '='


class Syntax(Enum):
    COMMA = ','
    L_PAREN = '('
    R_PAREN = ')'
    L_BRACKET = '['
    R_BRACKET = ']'
    L_CURLY_BRACE = '{'
    R_CURLY_BRACE = '}'


TokenKind = Enum(
    'TokenKind',
    ((m.name, m.value) for m in (*Literal, *BinOp, *Syntax))
)


class Token:
    def __init__(
        self,
        at: tuple[CharNo, Location],
        value: TokenValue,
        kind: TokenKind,
        match: re.Match,
    ) -> None:
        self.at = at
        self.value = value
        self.kind = kind
        self.match = match
        # print(match.groups())

    def __repr__(self):
        cls = type(self).__name__
        pairs = vars(self).items()
        stringified = tuple(
            (k, repr(v) if isinstance(v, str) else str(v))
            for k, v in pairs if k != 'match'
        )
        
        attrs = ', '.join(('='.join(pair) for pair in stringified))
        s = f"{cls}({attrs})"
        return s



class Lexer:

    rules = (
        # Data types
        (re.compile(
            r'^(?P<speech_char>["\'`]{3}|["\'`])'
            r'(?P<text>(?!(?P=speech_char)).*?)'
            r'(?P=speech_char)'
        ), Literal.LIT_STR),
        (re.compile(r'^\d*\.\d(\d|_)+'), Literal.LIT_FLOAT),
        (re.compile(r'^\d+'), Literal.LIT_INT),

        # Ignore
        (re.compile(r'^\#.*\n*'), Ignore.COMMENT),  # Skip comments
        (re.compile(r'^\s+'), Ignore.SPACE),  # Skip whitespace

        # Finds empty assignments:
        (re.compile(r'^\n+'), Ignore.NEWLINE),

        # Operators
        (re.compile(r'^='), BinOp.ASSIGN),
        (re.compile(r'^\.'), BinOp.ATTRIBUTE),

        # Syntax
        (re.compile(r'^\,'), Syntax.COMMA),
        (re.compile(r'^\('), Syntax.L_PAREN),
        (re.compile(r'^\)'), Syntax.R_PAREN),
        (re.compile(r'^\['), Syntax.L_BRACKET),
        (re.compile(r'^\]'), Syntax.R_BRACKET),
        (re.compile(r'^\{'), Syntax.L_CURLY_BRACE),
        (re.compile(r'^\}'), Syntax.R_CURLY_BRACE),

        # Symbols
        *tuple(dict.fromkeys(KEYWORDS, Symbol.KEYWORD).items()),
        (re.compile(r'^[a-zA-Z_]+'), Symbol.IDENTIFIER),
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
        self._lines = self._string.split('\n')
        self._file = file
        self._stream = StringIO(self._string)
        self._lookahead = None
        self._cursor = 0
        self._colno = 0
        self.eof = ''

    def error_leader(self) -> str:
        file = self._file if self._file is not None else ''
        return f"File {file!r}, line {self.lineno()}: "

    def error(self, token: str) -> NoReturn:
        pass

        e = ParserError(self.error_leader() + f"Unexpected token: {token!r}")

    def error_ctx(self) -> dict:
        return {'file': self._file, 'pos': self._cursor}

    def lineno(self) -> int:
        return self._string[:self._cursor].count('\n') + 1

    def curr_line(self) -> str:
        return self._lines[self.lineno - 1]

    def locate(self) -> Location:
        return (self.lineno(), self._colno)
    
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
        '''
        '''
        if self.at_eof():
            return self.eof

        s = self._string[self._cursor:]

        for test, kind in self.rules:
            m = re.match(test, s)

            if m is None:
                # Rule not matched
                continue

            if kind in Ignore:
                # Skip whitespace, comments
                self._cursor += len(m.group())
                self._colno = 0
                return self.get_token()

            # if kind is Literal.LIT_STR:
                # print(m.groups())

            tok = Token(
                value=m.group(),
                at=(self._cursor, self.locate()),
                kind=kind, match=m
            )
            self._cursor += len(m.group())
            self._colno += len(m.group())
            return tok

        s = s.split(None, 1)
        raise SyntaxError(self.error_leader() + f"Unexpected token: {s!r}")
        self.unexpected_token(s)


class ParserError(SyntaxError):
    def __init__(
        self,
        cause: str,
        file: PathLike = None,
        pos: tuple[int] = None,
        msg: str = "Unexpected token: ",
    ) -> None:
        # super().__init__(self, *args)
        self.cause = cause
        self.pos = pos
        self.msg = msg

    def __str__(self) -> str:
        # return self.error_leader() + 
        # if len(pos) == 1:
            # at = 
        file = self.file
        pos = self.pos
        msg = self.msg
        cause = self.cause
        return f"{file!r}, line {pos}: {msg}{cause!r}"


class BinaryExpression:
    def __init__(
        self,
        op: BinOp,
        left: Name | Constant,
        right ,#: Expression
    ) -> None:  # Bad types here
        self.op = op
        self.left = left
        self.right = right

    def parse(): pass


class Parser:
    def __init__(
        self,
    ) -> None: 
        self._lexer = Lexer()
        self._lookahead = None
        self.lookahead = self._lexer.get_token

    def parse_BinOp(self, curr: Token, nxt: Token) -> AST:
        if nxt.kind is BinOp.ASSIGN:
            if not curr.kind is Symbol.IDENTIFIER:
                raise SyntaxError(
                    self._lexer.error_leader()
                    + f"Cannot assign to non-identifier {curr.value!r}"
                )
            target = Name(id=curr.value)
            value = self.parse_expr()
            a = Assign(targets=[target], value=value)
            import ast
            print(ast.dump(a))
            return a

        
    def parse_expr(self) -> Tree:
        tree = []
        while True:

            # current_stmt = getattr(self, 'parse_' + tok.kind.name)(tok)
            # return tok.kind.parse
            # tree.append

            tok = self._lexer.get_token()
            if tok == self._lexer.eof:
                return tree
            nxt = self.lookahead()
            if nxt == self._lexer.eof:
                return tree


            if nxt.kind in BinOp:
                self.parse_BinOp(tok, nxt)
                # tree.append(eval_bin_op(tok))
##            elif tok.kind in BinOp:
##                self.eval_bin_op()
##            if tok.kind in BinOp:
##                self.eval_bin_op()
##            if tok.kind in BinOp:
##                self.eval_bin_op()


class Tree:
    pass

if __name__ == '__main__':
    # p = Parser()
    # t = Lexer(string=' '.join(sys.argv[1:]))
    t = Lexer()
    tokens = t.lex()
    # print(tokens)
    # return tokens
