import ast
import re
import sys

from ast import (
    AST,
    Constant,
    Dict,
    List,
    Module,
    Name,
)

from ast import Set 
from enum import Enum
from io import StringIO
from itertools import chain
from os import PathLike
from queue import LifoQueue
from typing import NamedTuple, NoReturn, TypeAlias


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]


FILE = 'test_config.conf'
EOF = ''
NEWLINE = '\n'


class Literal(Enum):
    FLOAT = 'FLOAT'
    INT = 'INT'
    STR = 'STR'
    TRUE = 'TRUE'
    FALSE = 'FALSE'

class Ignore(Enum):
    COMMENT = 'COMMENT'
    SPACE = 'SPACE'

class File(Enum):
    NEWLINE = NEWLINE
    EOF = EOF

class Symbol(Enum):
    IDENTIFIER = 'IDENTIFIER'
    KEYWORD = 'KEYWORD'

class Keyword(Enum):
    pass
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
    ((m.name, m.value) for m in (
        *Literal,
        *Symbol,
        *Syntax,
        *BinOp,
        *Ignore,
        *File,
    ))
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


class Grammar:
    '''
    mod = expr*
        | Assign(identifier target, expr? value)

    expr = Constant(constant value)
         | Dict(Assign*)
         | List(expr* elts)
         | Attribute(identifier value, identifier attr)

    constant = integer | float | string | boolean
    string = speech_char (text? speech_char{2}? text?)* speech_char
    speech_char = ['"`]{3} | ['"`]
    boolean = 'True' | 'False' | 'yes' | 'no'
    '''


class Lexer:
    STRING_CONCAT = True  # Concatenate neighboring strings
    SPEECH_CHARS = tuple('"\'`') + ('"""', "'''")

    _rules = (
        # Data types
        (re.compile(
            r'^(?P<speech_char>["\'`]{3}|["\'`])'
            r'(?P<text>(?!(?P=speech_char)).*?)*'
            r'(?P=speech_char)'
        ), Literal.STR),
        (re.compile(r'^\d*\.\d(\d|_)+'), Literal.FLOAT),
        (re.compile(r'^\d+'), Literal.INT),

        # Ignore
        (re.compile(r'^\#.*(?=\n*)'), Ignore.COMMENT),  # Skip comments
        (re.compile(r'^\n+'), File.NEWLINE),  # Finds empty assignments
        (re.compile(r'^\s+'), Ignore.SPACE),  # Skip all other whitespace

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
        (re.compile(r'^(true|yes)', re.IGNORECASE), Literal.TRUE),
        (re.compile(r'^(false|no)', re.IGNORECASE), Literal.FALSE),

        # Symbols
        *tuple(dict.fromkeys(KEYWORDS, Symbol.KEYWORD).items()),
        (re.compile(r'^[a-zA-Z_]+'), Symbol.IDENTIFIER),
        # (re.compile(r'^'), File.EOF),
    )

    def __init__(
        self,
        string: str = None,
        file: PathLike = None,
    ) -> None: 
        if string is None:
            if file is None:
                msg = "`string` parameter is required when `file` is not given"
                raise ValueError(msg)

            with open(file, 'r') as f:
                string = f.read()

        self._string = string
        self._lines = self._string.split('\n')
        self._file = file
        self._stream = StringIO(self._string)

        self._token_stack = LifoQueue()

        self._cursor = 0
        self._colno = 1
        self.eof = File.EOF

    def highlight_error(self, token: str, pos: Location) -> str:
        '''
        Highlight the part of a line occupied by a token.
        Return the original line surrounded by quotation marks followed
        by a line of spaces and carets that point to the token.

        :param token: The token to be highlighted
        :type token: :class:`str`

        :param pos: The token's position tuple, made of line number and
            column number
        :type pos: :class:`Location`
        '''
        lineno, colno = pos
        line = self._lines[lineno - 1].rstrip()
        # To dedent and preserve alignment, we need the column offset:
        offset = len(line) - len(line.lstrip())
        line = line.lstrip()

        highlight = " "  # Account for the repr quotation mark.
        highlight += (' ' * (colno - offset - 1)) + ('^' * len(token))
        return '\n'.join((repr(line), highlight))

    def error_leader(self, with_col: bool = False) -> str:
        '''
        '''
        file = self._file if self._file is not None else ''
        column = ', column ' + str(self._colno) if with_col else ''
        msg = f"File {file!r}, line {self.lineno()}{column}: "
        return msg

    def unexpected_token(self, token: str) -> SyntaxError:
        highlighted = self.highlight_error(token, self.coords())
        explained = self.error_leader() + f"Unexpected token: {token!r}"
        msg = '\n'.join((explained, highlighted))
        e = SyntaxError(msg)
        return e

    def unmatched_quote(self, token: str) -> SyntaxError:
        highlighted = self.highlight_error(token, self.coords())
        explained = self.error_leader() + f"Unmatched quote: {token!r}"
        msg = '\n'.join((explained, highlighted))
        e = SyntaxError(msg)
        return e

    def lineno(self) -> int:
        return self._string[:self._cursor].count('\n') + 1

    def curr_line(self) -> str:
        return self._lines[self.lineno - 1]

    def coords(self) -> Location:
        return (self.lineno(), self._colno)
    
    def lex(self, string: str = None) -> list[Token]:
        '''
        '''
        if string is not None:
            self._string = string

        tokens = []
        while True:
            tok = self.get_token()
            if tok.kind is self.eof:
                print("EOF  caught by Lexer.lex()")
                break
            tokens.append(tok)

        self.reset()
        return tokens

    def reset(self) -> None:
        '''
        Reset the cursor to the beginning of the lexer string.
        '''
        self._cursor = 0
        self._colno = 1

    def get_token(self) -> Token:
        '''
        '''
        # The stack will have contents after string concatenation.
        if not self._token_stack.empty():
            return self._token_stack.get()

        # Everything after and including the cursor position:
        s = self._string[self._cursor:]

        # Match against each rule:
        for test, kind in self._rules:
            m = re.match(test, s)

            if m is None:
                # This rule not matched; try the next one.
                continue

            tok = Token(
                value=m.group(),
                at=(self._cursor, self.coords()),
                kind=kind,
                match=m
            )

            # Update location:
            self._cursor += len(tok.value)
            self._colno += len(tok.value)
            if kind is File.NEWLINE:
                self._colno = 1

            if kind is Literal.STR:
                # Process strings by removing quotes:
                speech_char = tok.match.groups()[0]
                tok.value = tok.value.strip(speech_char)

                # Concatenate neighboring strings:
                if self.STRING_CONCAT:
                    while True:
                        maybe_str = self.get_token()
                        if maybe_str.kind in (*Ignore, File.NEWLINE):
                            continue
                        break

                    if maybe_str.kind is Literal.STR:
                        # Concatenate.
                        tok.value += maybe_str.value
                        return tok

                    else:
                        # Handle the next token separately.
                        self._token_stack.put(maybe_str)

            return tok

        else:
            if s is EOF:
                tok = Token(
                    value=s,
                    at=(self._cursor, self.coords()),
                    kind=self.eof,
                    match=None
                )
                # print(tok)

                return tok

            # If a token is not returned, prepare an error message:
            # print(repr(s))
            bad_token = s.split(None, 1)[0]
            if bad_token in self.SPEECH_CHARS:
                raise self.unmatched_quote(bad_token)
            raise self.unexpected_token(bad_token)


class Parser:
    '''
    Convert sequences of :class:`Token` to :class:`AST`.
    '''
    def __init__(
        self,
        string: str = None,
        *,
        file: PathLike = None,
    ) -> None:
        if string is None:
            if file is None:
                file = FILE

        self._string = string
        self._file = file
        self._lexer = Lexer(string, file)

    def tokens(self) -> list[Token]:
        return self._lexer.lex()

    def parse_as_dict(self) -> dict:
        '''
        Parse a config file and return its data as a :class:`dict`.
        '''
        i = Interpreter()
        mapping = {}
        for n in self.parse():
            mapping.update(i.visit(n))
        return mapping

    def parse(
        self,
        string: str = None,
        *,
        file: PathLike = None,
    ) -> AST: 
        '''
        Parse a string or file and return an AST using special grammar
        suited to config files.

        :param string: The string to parse. If `string` is ``None``,
            this method will open and read from `file`.
        :type string: :class:`str`

        :param file: The file to open and parse from. This is required
            if `string` is not set.
        :type file: :class:`PathLike`

        :returns: An AST using the grammar defined in :class:`Grammar`.
        :rtype: :class:`AST`
        '''
        new_data = False
        if string is None:
            if file is None:
                if self._string is None:
                    if self._file is None:
                        msg = (
                            "`string` parameter is required"
                            " when `file` is not given"
                        )
                        raise ValueError(msg)
                    file = self._file
                string = self._string

            else:
                new_data = True

            self._file = file
            self._string = string

        else:
            new_data = True

        if new_data:
            self._lexer = Lexer(self._string)

        body = []
        while True:
            try:
                parsed = self.parse_expr()
            except Exception as e:
                msg = self._lexer.error_leader()
                e.args = (msg + e.args[0],)
                raise e#.with_traceback(None)

            if parsed is self._lexer.eof:
                break

            body.append(parsed)

##            try:
##                body.append(parsed)
##                print(ast.dump(parsed))
##            except TypeError:
##                print(parsed)
##            print()

        return body

    def parse_expr(self, prev: Token = None) -> AST:
        '''
        Parse an expression and return its AST.
        This usually comes a key-value pair represented by a
        :class:`Dict` node.
        '''
        curr = self._lexer.get_token()

        if prev is None:
            # The first token in a file.
            return self.parse_expr(prev=curr)

        match curr.kind:

            case self._lexer.eof:
                return self._lexer.eof

            case Ignore.SPACE | Ignore.COMMENT:
                # Skip spaces and comments.
                node = self.parse_expr(prev)
                return node

            case File.NEWLINE:
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
                target = Name(prev.value)

                nxt = self.parse_expr(prev=curr)
                if (
                    isinstance(nxt, Token)
                    and nxt.kind in (File.NEWLINE, Syntax.COMMA)
                ):
                    value = Constant(None)

                elif isinstance(nxt, Token):
                    raise Exception((prev, curr, nxt))

                else:
                    value = nxt

                node = Dict([target], [value])

            case BinOp.ATTRIBUTE:
                if not prev.kind is Symbol.IDENTIFIER:
                    raise SyntaxError(f"{prev!r} doesn't do attributes")
                base = Name(prev.value)
                attr = self.parse_expr(prev=curr)
                node = Dict(keys=[base], values=[attr])

            case Syntax.L_CURLY_BRACE:
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
                node = reconciled

            case Syntax.L_BRACKET:
                elems = []
                while (node := self.parse_expr(prev=curr)):
                    if (
                        isinstance(node, Token)
                        and node.kind is Syntax.R_BRACKET
                    ):
                        break

                    elems.append(node)

                node = List(elts=elems)

            case Literal.STR:
                node = Constant(curr.value)

            case Literal.INT:
                node = Constant(int(curr.value))

            case Literal.FLOAT:
                node = Constant(float(curr.value))

            case Literal.TRUE:
                node = Constant(True)

            case Literal.FALSE:
                node = Constant(False)

            case _:
                node = curr

        return node


class Interpreter(ast.NodeVisitor):
    '''
    Convert ASTs to literal Python expressions.
    '''
    def visit_Constant(self, node):
        return node.value

    def visit_Dict(self, node):
        new_d = {}
        for k, v in zip(node.keys, node.values):
            new_d[self.visit(k)] = self.visit(v)
        return new_d

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Module(self, node, map: bool = True):
        return [self.visit(n) for n in node.body]

    def visit_Name(self, node):
        return node.id


if __name__ == '__main__':
    # l = Lexer(file=FILE)
    # print('\n'.join(str(t) for t in l.lex()))
    p = Parser()
    # for t in p.tokens:
        # print(t)
    # parsed = p.parse()
    print(p.parse_as_dict())

