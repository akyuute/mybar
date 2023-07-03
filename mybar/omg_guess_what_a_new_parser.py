import ast
import re
import sys

from ast import (
    AST,
    Assign,
    Attribute,
    BinOp,
    Constant,
    Dict,
    Expr,
    List,
    Module,
    Name,
    UnaryOp,
)

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
    LIT_TRUE = 'LIT_TRUE'
    LIT_FALSE = 'LIT_FALSE'

class Ignore(Enum):
    COMMENT = 'COMMENT'
    SPACE = 'SPACE'

class Newline(Enum):
    NEWLINE = 'NEWLINE'

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
    _rules = (
        # Data types
        (re.compile(
            r'^(?P<speech_char>["\'`]{3}|["\'`])'
            r'(?P<text>(?!(?P=speech_char)).*?)'
            r'(?P=speech_char)'
        ), Literal.LIT_STR),
        (re.compile(r'^\d*\.\d(\d|_)+'), Literal.LIT_FLOAT),
        (re.compile(r'^\d+'), Literal.LIT_INT),

        # Ignore
        (re.compile(r'^\#.*(?=\n*)'), Ignore.COMMENT),  # Skip comments
        (re.compile(r'^\n+'), Newline.NEWLINE),  # Finds empty assignments
        (re.compile(r'^\s+'), Ignore.SPACE),  # Skip whitespace

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

        # Booleans
        (re.compile(r'^(true|yes)', re.IGNORECASE), Literal.LIT_TRUE),
        (re.compile(r'^(false|no)', re.IGNORECASE), Literal.LIT_FALSE),

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
            if tok == self.eof:
                break
            tokens.append(tok)
        return tokens

    def get_token(self) -> Token:
        '''
        '''
        if self.at_eof():
            return self.eof

        s = self._string[self._cursor:]

        for test, kind in self._rules:
            m = re.match(test, s)

            if m is None:
                # Rule not matched.
                continue

            tok = Token(
                value=m.group(),
                at=(self._cursor, self.locate()),
                kind=kind, match=m
            )

            # Update location data:
            self._cursor += len(m.group())
            self._colno += len(m.group())
            if kind is Newline.NEWLINE:
                self._colno = 0

            return tok

        # If a token is not returned, prepare an error message:
        s = s.split(None, 1)[0]
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


class Parser:
    def __init__(
        self,
    ) -> None: 
        self._lexer = Lexer()
        self._tokens = self._lexer.lex()
        self._lexer = Lexer()

    def parse(self) -> Tree:
        '''
        '''
        body = []
        while not self._lexer.at_eof():
            parsed = self.parse_expr()
            if parsed is self._lexer.eof:
                break
            body.append(parsed)
            # print(ast.dump(parsed, indent=2))

        module = Module(body)
        print(ast.dump(module, indent=2))

    def parse_expr(self, prev: Token = None) -> AST:
        '''
        '''
        curr = self._lexer.get_token()

        if curr == self._lexer.eof:
            # print("EOF")
            return curr

        if prev is None:
            # The first token in a file.
            return self.parse_expr(prev=curr)

        match curr.kind:

            case self._lexer.eof:
                print("EOF")
                return curr

            case Ignore.SPACE | Ignore.COMMENT:
                # Skip spaces and comments.
                node = self.parse_expr(prev)
                return node

            case Newline.NEWLINE:
                if prev.kind is BinOp.ASSIGN:
                    # Notify of empty assignment, if applicable.
                    node = curr
                    return node
                else:
                    # Skip, otherwise.
                    node = self.parse_expr(prev=prev)
                    return node

            case Syntax.COMMA:
                # Continue, giving context.
                return self.parse_expr(prev=curr)

            case Symbol.IDENTIFIER:
                # Continue, giving context.
                if isinstance(prev, Token):
                    if prev.kind is Syntax.L_BRACKET:
                        # Inside a list.
                        return Name(curr.value)
                return self.parse_expr(prev=curr)

            case BinOp.ASSIGN:
                if not prev.kind is Symbol.IDENTIFIER:
                    raise SyntaxError(
                        self._lexer.error_leader()
                        + f"Cannot assign to non-identifier {prev.value!r}"
                    )
                # print(f"Assigning to {prev.value!r}")
                # target = Name(prev.value)
                target = prev.value

                nxt = self.parse_expr(prev=curr)
                # print(f"{nxt = }")
                if (
                    isinstance(nxt, Token)
                    and nxt.kind in (Newline.NEWLINE, Syntax.COMMA)
                ):
                    value = Constant(None)

                elif isinstance(nxt, Token):
                    raise Exception((prev, curr, nxt))

                else:
                    value = nxt

                node = Dict([target], [value])
                # rep = ast.dump(node)
                # print(rep)
                # print(f"{{{target} = {rep}\n}}")
                # return node

            case BinOp.ATTRIBUTE:
                if not prev.kind is Symbol.IDENTIFIER:
                # if not isinstance(prev, Name):
                    raise SyntaxError(f"{prev!r} doesn't do attributes")
                base = prev.value
                attr = self.parse_expr(prev=curr)
                node = Dict(keys=[base], values=[attr])

            case Syntax.L_CURLY_BRACE:
                # print()
                # print("Got assign for", prev.value)

                keys = []
                vals = []

                while (assign := self.parse_expr(prev=prev)):
                    if (
                        isinstance(assign, Token)
                        and assign.kind is Syntax.R_CURLY_BRACE
                    ):
                        break

                    elif isinstance(assign, Dict):

                        key = assign.keys[-1]
                        val = assign.values[-1]

                        if key not in keys:
                            keys.append(key)
                            vals.append(val)
                        else:
                            idx = keys.index(key)
                            vals[idx].keys.append(val.keys[-1])
                            vals[idx].values.append(val.values[-1])

                reconciled = Dict(keys, vals)
                # print(ast.dump(reconciled, indent=2))

                node = reconciled

            case Syntax.L_BRACKET:
                # print("Parsing list")
                elems = []
                while (node := self.parse_expr(prev=curr)):
                    if (
                        isinstance(node, Token)
                        and node.kind is Syntax.R_BRACKET
                    ):
                        break

                    elems.append(node)

                node = List(elts=elems)
                # rep = ast.dump(node, indent=2)
                # print(rep)
                # return node

            case Literal.LIT_STR:
                speech_char = curr.match.groups()[0]
                text = curr.value.strip(speech_char)
                node = Constant(text)
            case Literal.LIT_INT:
                node = Constant(int(curr.value))
            case Literal.LIT_FLOAT:
                node = Constant(float(curr.value))
            case Literal.LIT_TRUE:
                node = Constant(True)
            case Literal.LIT_FALSE:
                node = Constant(False)

            case _:
                node = curr

        return node


if __name__ == '__main__':
    p = Parser()
    # for t in p._tokens:
        # print(t)
    p.parse()
    # t = Lexer(string=' '.join(sys.argv[1:]))
    # t = Lexer()
    # tokens = t.lex()
    # print(tokens)
    # return tokens


