import ast
import re
import sys
from ast import (
    AST,
    Constant,
    Dict as _Dict,
    List,
    Name,
    NodeVisitor,
)
from enum import Enum
from io import StringIO
from itertools import chain
from os import PathLike
from queue import LifoQueue
from typing import NamedTuple, NoReturn, Self, TypeAlias, TypeVar

# from ._types import FileContents
FileContents = str

Token = TypeVar('Token')
Lexer = TypeVar('Lexer')


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]


FILE = 'test_config.conf'
EOF = ''
NEWLINE = '\n'

class Grammar:
    # I have no clue whether or not the following makes sense.
    '''
    mod = expr*
        | Assign(identifier target, expr? value)*

    expr = Constant(constant value)
         | Dict(Assign*)
         | List(expr* elts)
         | Attribute(identifier value, identifier attr)

    constant = integer | float | string | boolean
    string = speech_char (text? speech_char{2}? text?)* speech_char
    speech_char = ['"`]{3} | ['"`]
    boolean = 'True' | 'False' | 'yes' | 'no'
    '''


class Literal(Enum):
    FLOAT = 'FLOAT'
    INTEGER = 'INTEGER'
    STRING = 'STRING'
    TRUE = 'TRUE'
    FALSE = 'FALSE'

class Ignore(Enum):
    COMMENT = 'COMMENT'
    SPACE = 'SPACE'

class Misc(Enum):
    NEWLINE = NEWLINE
    EOF = EOF
    UNKNOWN = 'UNKNOWN'

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
        *Misc,
    ))
)


class ConfigError(SyntaxError):
    '''
    '''
    pass


class TokenError(ConfigError):
    '''
    '''
    def __init__(self, msg: str, *args, **kwargs) -> None:
        super().__init__(self)
        # Replicate the look of natural SyntaxError messages:
        self.msg = '\n' + msg

    @classmethod
    def with_highlighting(
        cls,
        tokens: Token | tuple[Token],
        msg: str,
        leader: str = None,
        with_col: bool = True,
        indent: int = 2
    ):
        '''
        Highlight the part of a line occupied by a token.
        Return the original line surrounded by quotation marks followed
        by a line of spaces and arrows that point to the token.

        :param tokens: The token or tokens to be highlighted
        :type tokens: :class:`Token` | tuple[:class:`Token`]

        :param msg: The error message to display after `leader`
            column number
        :type msg: :class:`str`

        :param leader: The error leader to use,
            defaults to that of the lexer of the first token
        :type leader: :class:`str`

        :param with_col: Display the column number of the token,
            defaults to ``True``
        :type with_col: :class:`bool`
        
        :param indent: Indent by this many spaces * 2,
            defaults to 2
        :type indent: :class:`int`

        :returns: A new :class:`TokenError` with a custom error message
        :rtype: :class:`TokenError`
        '''
        if isinstance(tokens, Token):
            tokens = (tokens,)

        token = tokens[0]
        lexer = token.lexer

        if indent is None:
            indent = 0
        dent = ' ' * indent

        if leader is None:
            if lexer is not None:
                leader = dent + token.error_leader(with_col)
            else:
                leader = ''

        dent = 2 * dent  # Double indent for following lines.
        line = lexer.get_line(token).rstrip()
        # To preserve alignment after lstrip(), we need the offset:
        lstrip_offset = len(line) - len(line.lstrip())
        space = ' ' * (token.colno - lstrip_offset - 1)
        highlight = dent + space
        line = dent + line.lstrip()


        if len(tokens) == 1:
            if token.kind is Literal.STRING:
                text = token.match.group()
            else:
                text = token.value

            highlight += '^' * len(text)
        
        else:
            # Highlight multiple tokens using all in the range:
            start = tokens[0].cursor
            end = tokens[-1].cursor

            try:
                # Reset the lexer since it's already passed our tokens:
                all_toks = lexer.reset().lex()
            except TokenError:
                all_toks = tokens

            all_between = [t for t in all_toks if start <= t.cursor <= end]

            for t in all_between:
                if t.kind is Misc.NEWLINE:
                    break

                if t.kind is Literal.STRING:
                    text = t.match.group()
                else:
                    text = t.value

                highlight += '^' * len(text)

        errmsg = leader + msg + '\n'.join(('', line, highlight))
        return cls(errmsg)


class ParseError(TokenError):
    '''
    '''
    pass


class StackTraceSilencer(SystemExit):
    '''
    '''
    pass


class Token:
    '''
    '''
    __slots__ = (
        'at',
        'value',
        'kind',
        'match',
        'cursor',
        'lineno',
        'colno',
        'lexer',
        'file',
    )

    def __init__(
        self,
        at: tuple[CharNo, Location],
        value: TokenValue,
        kind: TokenKind,
        match: re.Match,
        lexer: Lexer = None,
        file: PathLike = None,
    ) -> None:
        self.at = at
        self.value = value
        self.kind = kind
        self.match = match
        self.cursor = at[0]
        self.lineno = at[1][0]
        self.colno = at[1][1]
        self.lexer = lexer
        self.file = file

    def __repr__(self):
        cls = type(self).__name__
        ignore = ('cursor', 'lineno', 'colno', 'lexer')
        pairs = (
            (k, getattr(self, k)) for k in self.__slots__
            if k not in ignore
        )
        stringified = tuple(
            # Never use repr() for `Enum` instances:
            (k, repr(v) if isinstance(v, str) else str(v))
            for k, v in pairs if k != 'match'
        )
        
        attrs = ', '.join(('='.join(pair) for pair in stringified))
        s = f"{cls}({attrs})"
        return s

    def error_leader(self, with_col: bool = False) -> str:
        '''
        '''
        file = f"File {self.file}, " if self.file is not None else ""
        column = ', column ' + str(self.colno) if with_col else ""
        msg = f"{file}line {self.lineno}{column}: "
        return msg

    def coords(self) -> Location:
        return (self.lineno, self.colno)
    
    def check_expected_after(
        self,
        prev: Token,
        expected: TokenKind | tuple[TokenKind],
        msg: str = None,
        error: bool = True
    ) -> bool | NoReturn:
        '''
        '''
        if isinstance(expected, tuple):
            invalid = (
                isinstance(prev, Token)
                and prev.kind not in expected
            )
        else:
            invalid = (
                isinstance(expected, Token)
                and prev.kind is not expected
            )

        if invalid:
            if error:
                if msg is None:
                    msg = f"Unexpected token: {self.value!r}"
                raise TokenError.with_highlighting(self, msg)

            return False
        return True


class Lexer:
    STRING_CONCAT = True  # Concatenate neighboring strings
    SPEECH_CHARS = tuple('"\'`') + ('"""', "'''")

    _rules = (
        # Data types
        (re.compile(
            r'^(?P<speech_char>["\'`]{3}|["\'`])'
            r'(?P<text>(?!(?P=speech_char)).*?)*'
            r'(?P=speech_char)'
        ), Literal.STRING),
        (re.compile(r'^\d*\.\d[\d_]*'), Literal.FLOAT),
        (re.compile(r'^\d+'), Literal.INTEGER),

        # Ignore
        ## Skip comments
        (re.compile(r'^\#.*(?=\n*)'), Ignore.COMMENT),
        ## Finds empty assignments:
        (re.compile(r'^' + NEWLINE + r'+'), Misc.NEWLINE),
        ## Skip all other whitespace:
        (re.compile(r'^[^' + NEWLINE + r'\S]+'), Ignore.SPACE),

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
        # (re.compile(r'^'), Misc.EOF),
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

        self._cursor = 0  # 0-indexed
        self._lineno = 1  # 1-indexed
        self._colno = 1  # 1-indexed
        self.eof = Misc.EOF

    def lineno(self) -> int:
        return self._string[:self._cursor].count('\n') + 1

    def curr_line(self) -> str:
        return self._lines[self._lineno - 1]

    def get_line(self, lookup: int | Token) -> str:
        # lineno = (lookup.at[1][0] if isinstance(lookup, Token) else lookup)
        if isinstance(lookup, Token):
            lineno = lookup.at[1][0]
        else:
            lineno = lookup
        return self._lines[lineno - 1]

    def coords(self) -> Location:
        return (self._lineno, self._colno)
    
    def lex(self, string: str = None) -> list[Token]:
        '''
        '''
        if string is not None:
            self._string = string

        tokens = []
        try:
            while True:
                tok = self.get_token()
                if tok.kind is self.eof:
                    break
                tokens.append(tok)
        except TokenError as e:
            import traceback
            traceback.print_exc(limit=1)
            raise 

        self.reset()
        return tokens

    def reset(self) -> Self:
        '''
        Reset the cursor to the beginning of the lexer string.
        '''
        self._cursor = 0
        self._colno = 1
        return self

    def error_leader(self, with_col: bool = False) -> str:
        '''
        '''
        file = self._file if self._file is not None else ''
        column = ', column ' + str(self._colno) if with_col else ''
        msg = f"File {file!r}, line {self._lineno}{column}: "
        return msg

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
                at=(self._cursor, (self._lineno, self._colno)),
                kind=kind,
                match=m,
                lexer=self,
                file=self._file
            )

            # Update location:
            self._cursor += len(tok.value)
            self._colno += len(tok.value)
            if kind is Misc.NEWLINE:
                self._lineno += len(m.group())
                self._colno = 1

            if kind is Literal.STRING:
                # Process strings by removing quotes:
                speech_char = tok.match.groups()[0]
                tok.value = tok.value.strip(speech_char)

                # Concatenate neighboring strings:
                if self.STRING_CONCAT:
                    while True:
                        maybe_str = self.get_token()
                        if maybe_str.kind in (*Ignore, Misc.NEWLINE):
                            continue
                        break

                    if maybe_str.kind is Literal.STRING:
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
                    at=(self._cursor, (self._lineno, self._colno)),
                    kind=self.eof,
                    match=None,
                    lexer=self,
                    file=self._file
                )

                return tok

            # If a token is not returned, prepare an error message:
            bad_value = s.split(None, 1)[0]
            bad_token = Token(
                value=bad_value,
                at=(self._cursor, (self._lineno, self._colno)),
                kind=Misc.UNKNOWN,
                match=None,
                lexer=self,
                file=self._file
            )
            try:
                if bad_token in self.SPEECH_CHARS:
                    msg = f"Unmatched quote: {bad_value!r}"
                else:
                    msg = f"Unexpected token: {bad_value!r}"

                raise TokenError.with_highlighting(bad_token, msg)

            except TokenError as e:
                # Avoid recursive stack traces with hundreds of frames:
                import traceback
                traceback.print_exc(limit=1)
                raise StackTraceSilencer(1)  # Sorry...


class Dict(_Dict):
    '''
    A custom ast.Dict class with an instance attribute for logging
    whether or not a mapping encloses other mappings.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._enclosing = False

    def __repr__(self) -> str:
        cls = type(self).__name__
        enc = self._enclosing
        # keys = [ast.dump(k) for k in self.keys]
        # values = [ast.dump(v) for v in self.values]
        keys = [k if isinstance(k, str) else k.id for k in self.keys]
        # keys = [k if isinstance(k, str) else ast.dump(k) for k in self.keys]
        # values = [v if isinstance(v, str) else ast.dump(v) for v in self.keys]
        r = f"{cls}({enc=}, {keys=})"
        # r = f"{cls}({keys=}, {values=})"
        return r


class Interpreter(NodeVisitor):
    '''
    Convert ASTs to literal Python expressions.
    '''
    def visit_Constant(self, node):
        return node.value

    def visit_Dict(self, node):
        # print()
        # print(ast.dump(node))


        new_d = {}

        for key, value in zip(node.keys, node.values):
            # print()
            k = self.visit(key)
            v = self.visit(value)
            # print(ast.dump(key), ast.dump(value))
            # new_d[self.visit(key)] = self.visit(v)

                # if maybe_assign.pop(self.__NESTED, False):
                    # 'kwargs.foo'
                    # base, attr = maybe_assign.popitem()

            if key not in new_d:
                new_d[k] = v
            else:
                new_d[k].update(v)

##        new_d = {}
##        # print(
##        for key, val in zip(node.keys, node.values):
##            print()
##            print(ast.dump(key), ast.dump(val))
##            new_d[self.visit(key)] = self.visit(val)


        return new_d

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Name(self, node):
        return node.id

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
        self._previous_token = None

    def tokens(self) -> list[Token]:
        return self._lexer.lex()

    def parse_as_dict(self, tree = None) -> dict:
        '''
        Parse a config file and return its data as a :class:`dict`.
        '''
        if tree is None:
            self._lexer.reset()
            tree = self.parse()

        i = Interpreter()
        mapping = {}
        for n in tree:
        # for n in self.parse():
            if not isinstance(n, Dict):
                # Each expression must map one thing to another.
                # This expression breaks the rules.
                note = (
                    f"{n._token.kind.value} cannot be at"
                    f" the start of an expression."
                )
                msg = f"Invalid syntax: {n._token.match.group()!r} ({note})"
                raise ParseError.with_highlighting(n._token, msg)

            parsed = i.visit(n)
            # print(parsed)
            mapping.update(parsed)
        return mapping

    def parse(
        self,
        string: str = None,
        *,
        file: PathLike = None,
    ) -> list[AST]:
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
        try:
            while True:
                parsed = self.parse_expr()

                if parsed is self._lexer.eof:
                    break

                if isinstance(parsed, Dict):
                    if any(isinstance(v, Dict) for v in parsed.values):
                        parsed._enclosing = True
                    # print(repr(parsed))

                body.append(parsed)
                # print([repr(d_) for d in parsed.values for d_ in d.values])

        except TokenError as e:
            import traceback, sys
            traceback.print_exc(limit=1)
            sys.exit(1)

        return body

    def parse_expr(self, prev: Token = None) -> AST:
        '''
        Parse an expression and return its AST.
        This usually comes a key-value pair represented by a
        :class:`Dict` node.
        '''
        curr = self._lexer.get_token()

        if prev is None:
            # The first token in an expression.
            if curr.kind is Syntax.COMMA:
                msg = "Invalid syntax: Comma used outside [] or {}:"
                raise ParseError.with_highlighting(curr, msg)

            if curr.kind in Syntax:
                msg = f"Unmatched {curr.value!r}"
                raise ParseError.with_highlighting(curr, msg)

            # if curr.kind in BinOp:
            if curr.kind in (*BinOp, ): #*Syntax, *Literal):
                # One cannot begin an expression with any of these.
                # msg = "Invalid syntax"
                line = curr.lexer.get_line(curr).lstrip()
                msg = f"Invalid syntax:" #{line!r}"
                raise ParseError.with_highlighting(curr, msg)

            # Advance to the next expression:
            self._previous_token = curr
            return self.parse_expr(prev=curr)

        self._previous_token = curr

        match curr.kind:

            case self._lexer.eof:
                if prev.value in '[{(':
                    msg = f"Unterminated {prev.value!r}"
                    raise ParseError.with_highlighting(prev, msg)

                return self._lexer.eof

            case Ignore.SPACE | Ignore.COMMENT:
                # Skip spaces and comments.
                node = self.parse_expr(prev)
                return node

            case Misc.NEWLINE:
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
                    if prev.kind is Symbol.IDENTIFIER:
                        msg = f"Unexpected identifier: {curr.value!r}"
                        raise ParseError.with_highlighting(curr, msg)

                    # Remove once scoped self-reference is available:
                    if prev.kind is BinOp.ASSIGN:
                        note = "cannot assign an identifier"
                        msg = f"Invalid assignment ({note}):"
                        raise ParseError.with_highlighting((prev, curr), msg)

                    if prev.kind is Syntax.L_BRACKET:
                        # Consecutive identifiers are valid inside lists.
                        return Name(curr.value)

                nxt = self.parse_expr(prev=curr)
                # if isinstance(nxt, Dict):
                return nxt

            case BinOp.ASSIGN:
                # print(prev)
                if isinstance(prev, Token):
                    # Only ever assign to identifiers:
                    if prev.kind is not Symbol.IDENTIFIER:
                        msg = "Invalid syntax:"
                        raise ParseError.with_highlighting((prev, curr), msg)
                        
                target = Name(prev.value)

                val = self.parse_expr(prev=curr)
                if isinstance(val, Token):
                    # Handle empty assignments:
                    if val.kind in (Misc.NEWLINE, Syntax.COMMA):
                        value = Constant(None)

                    # Unreachable?
                    else:
                        msg = f"Invalid syntax for assignment: {val.value!r}"
                        # raise ParseError.with_highlighting(val, msg)
                        raise ParseError.with_highlighting((prev, curr, val), msg)

                else:
                    value = val

                node = Dict([target], [value])
                if isinstance(prev, Token) and prev.kind is Syntax.L_CURLY_BRACE:
                    print("Found an assign following a {")
                    node._enclosing = True

            case BinOp.ATTRIBUTE:
                curr.check_expected_after(prev, (Symbol.IDENTIFIER,))
                base = Name(prev.value)
                attr = self.parse_expr(prev=curr)
                node = Dict(keys=[base], values=[attr])
                # node._enclosing = True

            case Syntax.L_CURLY_BRACE:
                keys = []
                vals = []

                # Gather assignments, which come in the form of `Dict`:
                while True:
                    # print(keys)
                    # print([ast.dump(v) for v in vals])
                    assign = self.parse_expr(prev=curr)
                    if isinstance(assign, Token):
                        if assign.kind is Syntax.R_CURLY_BRACE:
                            break

                    elif isinstance(assign, Dict):
                        if not assign.keys:
                            continue
                        
                        # print(f"{ast.dump(assign) = }")
                        # Assignments only have one key and one value:
                        key = assign.keys[-1].id
                        val = assign.values[-1]
                        # print(ast.dump(val))

                        if key not in keys:
                            keys.append(key)
                            vals.append(val)

                        else:
                            # Make nested dicts:
                            # if isinstance(val, Dict):
                                

                            idx = keys.index(key)
                            # print(key, idx)
                            # vals[idx].values.append(val)
                            # print([ast.dump(v) for v in vals[idx].values])
                            vals[idx].keys.append(val.keys[-1])
                            vals[idx].values.append(val.values[-1])
                            # vals[idx].keys.append(val.keys[idx])
                            # vals[idx].values.append(val.values[idx])
                            # print([ast.dump(v) for v in vals[idx].values])

                # Put all the assignments together:
                reconciled = Dict([Name(k) for k in keys], vals)

                # Is the enclosing assignment made using '='?
                if prev.kind is Symbol.IDENTIFIER:
                    # No. ('foo {bar=1, baz=2, ...')
                    # Wrap ourselves in Dict as if assigning to 'foo':
                    node = Dict(keys=[Name(prev.value)], values=[reconciled])
                    node._enclosing = True
                else:
                    # Yes. ('foo = {bar=1, baz=2, ...')
                    # Return ourselves to the '=':
                    node = reconciled

            case Syntax.R_CURLY_BRACE:
                curr.check_expected_after(
                    prev, (Syntax.L_CURLY_BRACE, Syntax.COMMA)
                )
                node = curr

            case Syntax.L_BRACKET:
                b = Syntax.R_BRACKET
                elems = []
                # Gather elements.
                while True:
                    node = self.parse_expr(prev=curr)
                    if isinstance(node, Token) and node.kind is b:
                        break

                    elems.append(node)

                reconciled = List(elts=elems)

                # Is the enclosing assignment made using '='?
                if prev.kind is Symbol.IDENTIFIER:
                    # No. ('foo [bar, baz, ...')
                    # Wrap ourselves in Dict as if assigning to 'foo':
                    node = Dict(keys=[Name(prev.value)], values=[node])
                else:
                    # Yes. ('foo = {bar=1, baz=2, ...')
                    # Return ourselves to the '=':
                    node = reconciled

                # node._enclosing = True
                # print(repr(node))
                # print()

            case Syntax.R_BRACKET:
                curr.check_expected_after(
                    prev, (Syntax.L_BRACKET, Syntax.COMMA)
                )
                node = curr

            # Return literals as Constants:
            case Literal.STRING:
                node = Constant(curr.value)

            case Literal.INTEGER:
                node = Constant(int(curr.value))

            case Literal.FLOAT:
                node = Constant(float(curr.value))

            case Literal.TRUE:
                node = Constant(True)

            case Literal.FALSE:
                node = Constant(False)

            case _:
                node = curr

        if isinstance(node, AST):
            node._token = curr
        return node

class Uninterpreter(NodeVisitor):
    '''
    Convert ASTs back into string representations.
    '''
    indent = 4 * ' '

    def stringify(self, tree: list[AST]) -> FileContents:
        for n in tree:
            # print(f"{ast.dump(n) = }")
            string = self.visit(n)
            # print(repr(string))
            # print(string)
        return '\n'.join((self.visit(n) for n in tree))

    def visit_Dict(self, node):
        print()
        # print(ast.dump(node))
        # print(node._enclosing)
        # print(repr(node))
        # print("Got dict")
        # print(ast.dump(node))

        if node._enclosing:
            print("Enclosing:")
            # print([ast.dump(v) for v in node.values])
            # print("Targets:", [k.id for k in node.keys])

            target = self.visit(node.keys[-1])  # There will only be one.
            print(target)
            # assignments = [self.visit(v) for v in node.values]
            assignments = self.visit(node.values[-1])
            # print(assignments)
            # joined = ('\n' + self.indent).join(assignments)
            string = f"{target} {{\n{self.indent}{assignments}\n}}"
            # string = f"{target} {{\n{self.indent}{joined}\n}}"
            return string

        assignments = []
        print("Not enclosing")
        for k, v in zip(node.keys, node.values):
            print(ast.dump(node))
            # print(v)
            key = self.visit(k)
            value = self.visit(v)
            print((v))
            print(key)
            print(key, value)

            if isinstance(v, Dict):
                # Kwargs, for example, gotten by attributes.
                print("Yes a dict")
                # vals = [f"{key}.{val}" for val in value]


                vals = []
                for val in v.values:
                    attr = self.visit(val) if isinstance(val, Dict) else value
                    vals.append(f"{key}.{attr}")
                    print(vals)
                # vals = [f"{key}.{value}" for val in v.values]


                # attribute = f"{key}.{value}"
                attributes = ('\n' + self.indent).join(vals)
                # print(string)
                # assignments.append(attribute)
                assignments.append(attributes)

            else:
                print("Not a dict:")
                print(key, value)
                # print(ast.dump(value))
                # Plain assignments to a simplex identifier.
                assign = f"{key} = {value}"
                assignments.append(assign)

##            else:
##                print("Not enclosing:")
####                for k, v in zip(node.keys, node.values):
####                    # print(v)
####                    key = self.visit(k)
####                    value = self.visit(v)
####                    # print(ast.dump(v))
##                print(f"{key = }")
##                print(f"{value = }")
##
##                if value.startswith("{"):
##                    string = f"{key} {value}"
##                else:
##                    string = f"{key} = {value}"


            print(f"{assignments = }")

        joined = ('\n' + self.indent).join(assignments)
        string = f"{self.indent}{joined}\n}}"
        string = joined
        print(string)
        return string
        # return assignments

        # return string

        # return string

##        if node._enclosing:

##            # print("Enclosing:")
##            # print([ast.dump(v) for v in node.values])
##            # keys = node.values[-1].keys
##            # values = node.values[-1].values
##
##            for k, v in zip(node.keys, node.values):
##                # print(v)
##                key = self.visit(k)
##                value = self.visit(v)
##
##                if isinstance(v, Dict):
##                    # Kwargs, for example, gotten by attributes.
##                    attribute = f"{key}.{value}"
##                    # print(string)
##                    assignments.append(attribute)
##
##                else:
##                    # print("Not a dict:")
##                    # print(key, value)
##                    # print(ast.dump(value))
##                    # Plain assignments to a simplex identifier.
##                    assign = f"{key} = {value}"
##                    assignments.append(assign)
##
##                print(f"{assignments = }")
##
##            # print("Targets:", [k.id for k in node.keys])
##            joined = ('\n' + self.indent).join(assignments)
##            string = f"{{\n{self.indent}{joined}\n}}"
##
##            # return string
##
##        else:
##            print("Not enclosing:")
##            for k, v in zip(node.keys, node.values):
##                # print(v)
##                key = self.visit(k)
##                value = self.visit(v)
##                # print(ast.dump(v))
##                print(f"{key = }")
##                print(f"{value = }")
##                if value.startswith("{"):
##                    string = f"{key} {value}"
##                else:
##                    string = f"{key} = {value}"

        return string

    def visit_List(self, node):
        elems = tuple(self.visit(e) for e in node.elts)
        if len(elems) > 3:
            joined = ('\n' + self.indent).join(elems)
            string = f"[\n{self.indent}{joined}\n]"
        else:
            joined = ', '.join(elems)
            string = f"[{joined}]"
        return string

    def visit_Name(self, node):
        string = node.id
        return string

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, str):
            return repr(val)
        if val is None:
            return ""
        return str(val)


#TODO: Finish dict unparse implementation!
class Unparser:
    '''
    Convert Python dictionaries to ASTs.
    '''
    def unparse(self, mapping: dict) -> list[AST]:
        if not isinstance(mapping, dict):
            # print(mapping)
            return mapping

        assignments = []
        # keys = []
        # vals = []

        # return [Dict([Name(k)], [self.unparse_node(v)]) for k, v in mapping.items()]

        for key, val in mapping.items():
            # keys.append(key)
            # vals.append()
            node = Dict([Name(key)], [self.unparse_node(val)])
            # node._enclosing = True
            assignments.append(node)
        return assignments

    def unparse_node(self, thing) -> list[AST]:
        '''
        '''
        if isinstance(thing, dict):
            keys = []
            vals = []

            # print(thing.items())
            for key, val in thing.items():
                
                # print(self.unparse_node(val))

                if key not in keys:
                    # print(f"{key} not in {keys}")
                    keys.append(key)
                    vals.append(self.unparse_node(val))
                else:
                    # print(f"{key} in {keys}")
                    idx = keys.index(key)
                    vals[idx].keys.append(v.keys[-1])
                    vals[idx].values.append(v.values[-1])

##                subkeys = []
##                subvals = []
##                for k, v in val.items():
##
##                    if k not in subkeys:
##                        print(f"{k} not in {subkeys}")
##                        subkeys.append(k)
##                        subvals.append(self.unparse(v))
##                    else:
##                        print(f"{k} in {subkeys}")
##                        idx = keys.index(k)
##                        subvals[idx].values.append(self.unparse(v))

                # print(subkeys, subvals)
                # print(subkeys, [ast.dump(v) for v in vals])

            # keys.append(key)
            # vals.append(Dict(subkeys, subvals))

            node = Dict([Name(k) for k in keys], vals)
            # if any(isinstance(v, Dict) for v in vals):
                # node._enclosing = True
                # print("Enclosing", ast.dump(node))
            # print(keys, vals)
            # node = Dict(keys, vals)

            # print([ast.dump(v) for v in vals])
            # node = Dict([k], v)
            # print(ast.dump(node))
            # print()
            return node

            # vals.append(node)
            


        elif isinstance(thing, list):
            # vals = [Constant(v) for v in thing]
            values = [Name(v) for v in thing]
            # node = Dict([target], [List(values)])
            return List(values)

        elif isinstance(thing, (int, float, str)):
            return Constant(thing)
            # node = Dict([target], [Constant(thing)])

        elif thing in (None, True, False):
            return Constant(thing)
            # node = Dict([target], [Constant(thing)])

        # keys.append(key)
        # vals.append(node)
        # assignments.append(node)

        # node = Dict(keys, vals)
        # print(f"{ast.dump(node) = }")
        # print([ast.dump(k) for k in keys])
        # print([ast.dump(v) for v in vals])
        # print(assignments)

        return node
        # print(vals)
        # assignments.append(Dict(keys, vals))
        # return assignments

    def unparse_from_dict(self, mapping: dict) -> FileContents:
        '''
        '''
        d = Uninterpreter()
        unparsed = self.unparse(mapping)
        for d in unparsed:
            print(ast.dump(d, indent=2))
        # return d.stringify(self.unparse(mapping))




if __name__ == '__main__':
    # l = Lexer(file=FILE)
    # l.lex()
    # print('\n'.join(str(t) for t in l.lex() if not t.kind in (*Ignore, Misc.NEWLINE)))
    p = Parser()
    # print(ast.dump(p.parse()[0]))
    # print(p.parse_as_dict())
##    for n in p.parse():
##        print(ast.dump(n, ))
    # p._lexer.reset()
    # print()
    # print()
    # print()
##    print("From config file:")
    parsed = p.parse_as_dict()
##    print(parsed)
##    print()
####    for k, v in parsed.items():
####        print(v)
####        print(f"<{k}: {ast.dump(Unparser().unparse_node(v))}>")
##    print("From unparsing and reparsing the above")
##    print("(should be exactly the same):")
    unparsed = Unparser().unparse(parsed)
    unreparsed = p.parse_as_dict(unparsed)
##    print(unreparsed)
    # for n in Unparser().unparse(parsed):
        # print(ast.dump(n))
    # for n in Unparser().unparse(parsed):
        # print(ast.dump(n, ))
    # print(Unparser().unparse_from_dict(p.parse_as_dict()))

    p._lexer.reset()

##    print()
##    # print([repr(d_) for d in p.parse() for d_ in d.values])
##    # print([repr(d_) for d in unparsed for d_ in d.values])
##    print([(d._enclosing, ast.dump(d)) for d in p.parse()])
##    p._lexer.reset()
##    print([(d_._enclosing, ast.dump(d_)) for d in p.parse() for d_ in d.values])
##    print()
##    print([(d._enclosing, ast.dump(d)) for d in unparsed])
##    print([(d_._enclosing, ast.dump(d_)) for d in unparsed for d_ in d.values])
##    print()
##    print(Uninterpreter().stringify(unparsed))
    print(Uninterpreter().stringify(p.parse()))
    # print(Uninterpreter().stringify(unparsed))
##    p._lexer.reset()
##    print()
##    print(Uninterpreter().stringify(Unparser().unparse(p.parse_as_dict())))
##
