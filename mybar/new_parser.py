import ast
import re
import string
import sys
from ast import (
    AST,
    Assign,
    Attribute,
    Constant,
    Dict,
    List,
    Module,
    Name,
    NodeVisitor,
)
from enum import Enum
from os import PathLike
from queue import LifoQueue
from collections.abc import Mapping, MutableMapping, Sequence, MutableSequence
from typing import Any, NamedTuple, NoReturn, Self, TypeAlias, TypeVar

# from ._types import FileContents
FileContents = str

Token = TypeVar('Token')
Lexer = TypeVar('Lexer')


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]


FILE = 'mybar/test_config.conf'
EOF = ''
NEWLINE = '\n'
NOT_IN_IDS = string.punctuation.replace('_', '\s')


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

class Newline(Enum):
    NEWLINE = repr(NEWLINE)

class EndOfFile(Enum):
    EOF = repr(EOF)

class Unknown(Enum):
    UNKNOWN = 'UNKNOWN'

class Symbol(Enum):
    IDENTIFIER = 'IDENTIFIER'
    KEYWORD = 'KEYWORD'

class Keyword(Enum):
    pass
KEYWORDS = ()

class BinOp(Enum):
    ASSIGN = '='
    ATTRIBUTE = '.'
    ADD = '+'
    SUBTRACT = '-'

class Syntax(Enum):
    COMMA = ','
    L_PAREN = '('
    R_PAREN = ')'
    L_BRACKET = '['
    R_BRACKET = ']'
    L_CURLY_BRACE = '{'
    R_CURLY_BRACE = '}'

AssignEvalNone = tuple((*Newline, *EndOfFile, *Unknown, ))

TokenKind = Enum(
    'TokenKind',
    ((k.name, k.value) for k in (
        *Literal,
        *Symbol,
        *Syntax,
        *BinOp,
        *Ignore,
        *AssignEvalNone,
    ))
)


class ConfigError(SyntaxError):
    '''
    Base exception for errors related to config file parsing.
    '''
    pass


class TokenError(ConfigError):
    '''
    An exception affecting individual tokens and their arrangement.
    '''
    def __init__(self, msg: str, *args, **kwargs) -> None:
        super().__init__(self)
        # Replicate the look of natural SyntaxError messages:
        self.msg = '\n' + msg

    @classmethod
    def hl_error(
        cls,
        tokens: Token | tuple[Token],
        msg: str,
        with_col: bool = True,
        leader: str = None,
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

        :param with_col: Display the column number of the token,
            defaults to ``True``
        :type with_col: :class:`bool`
        
        :param leader: The error leader to use,
            defaults to that of the lexer of the first token
        :type leader: :class:`str`

        :param indent: Indent by this many spaces * 2,
            defaults to 2
        :type indent: :class:`int`

        :returns: A new :class:`TokenError` with a custom error message
        :rtype: :class:`TokenError`
        '''
        if isinstance(tokens, Token):
            tokens = (tokens,)

        first_tok = tokens[0]
        lexer = first_tok.lexer

        if indent is None:
            indent = 0
        dent = ' ' * indent

        if leader is None:
            if lexer is not None:
                leader = dent + first_tok.error_leader(with_col)
            else:
                leader = ""

        max_len = 100
        break_line = "\n" + dent if len(leader + msg) > max_len else ""
        dent = 2 * dent  # Double indent for following lines.
        highlight = ""

        if len(tokens) == 1:
            line_bridge = " "
            line = lexer.get_line(first_tok)
            if first_tok.kind is Literal.STRING:
                text = first_tok.match_repr()
            else:
                text = first_tok.value
            between = tokens

        else:
            # Highlight multiple tokens using all in the range:
            start = tokens[0].cursor
            end = tokens[-1].cursor
            line_bridge = " "

            try:
                # Reset the lexer since it's already passed our tokens:
                all_toks = lexer.reset().lex()
            except TokenError:
                all_toks = tokens

            between = tuple(t for t in all_toks if start <= t.cursor <= end)

            if any(t.kind is Newline.NEWLINE for t in between):
                # Consolidate multiple lines:
                with_dups = (
                    lexer.get_line(t) for t in between if t.kind not in Ignore
                )
                lines = dict.fromkeys(with_dups)
                # Don't count line breaks twice:
                lines.pop('', None)
                line = line_bridge.join(lines)
            else:
                line = lexer.get_line(first_tok)

        # Work out the highlight line:
        for t in between:
            match t.kind:
                case Ignore.COMMENT | Ignore.SPACE:
                    if t is between[-1]:
                        highlight += '^'
                    continue
                    highlight += " " * len(t.value)
                    continue

                case t.kind if t.kind in Ignore:
                    continue

                case Newline.NEWLINE:
                    if t is between[-1]:
                        highlight += '^'
                    continue

                case Literal.STRING:
                    # match_repr() contains the quotation marks:
                    token_length = len(t.match_repr())

                case _:
                    token_length = len(t.value)

            highlight += '^' * token_length

        # Determine how far along the first token is in the line:
        line_start = len(line) - len(line.lstrip())
        if between[-1].kind is Newline.NEWLINE:
            line_end = len(line) - len(line.rstrip())
            line_start += line_end
        tok_start_distance = first_tok.colno - line_start - 1
        offset = ' ' * tok_start_distance
        highlight = dent + offset + highlight
        line = dent + line.strip()

        errmsg = leader + break_line + msg + '\n'.join(('', line, highlight))
        return cls(errmsg)


class ParseError(TokenError):
    '''
    Exception raised during file parsing operations.
    '''
    pass


class StackTraceSilencer(SystemExit):
    '''
    Raised to skip printing a stack trace in deeply recursive functions.
    '''
    pass


class Token:
    '''
    Represents a single lexical word in a config file.

    :param at: The token's location
    :type at: tuple[:class:`CharNo`, :class:`Location`

    :param value: The literal text making up the token
    :type value: :class:`str`

    :param kind: The token's distict kind
    :type kind: :class:`TOkenKind`

    :param matchgroups: The value gotten by re.Match.groups() when
        making this token
    :type matchgroups: tuple[:class:`str`]

    :param lexer: The lexer used to find this token, optional
    :type lexer: :class:`Lexer`

    :param file: The file from which this token came, optional
    :type file: :class:`PathLike`
    '''
    __slots__ = (
        'at',
        'value',
        'kind',
        'matchgroups',
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
        matchgroups: tuple[str],
        lexer: Lexer = None,
        file: PathLike = None,
    ) -> None:
        self.at = at
        self.value = value
        self.kind = kind
        self.matchgroups = matchgroups
        self.cursor = at[0]
        self.lineno = at[1][0]
        self.colno = at[1][1]
        self.lexer = lexer
        self.file = file

    def __repr__(self):
        cls = type(self).__name__
        ignore = ('matchgroups', 'cursor', 'lineno', 'colno', 'lexer', 'file')
        pairs = (
            (k, getattr(self, k)) for k in self.__slots__
            if k not in ignore
        )
        stringified = tuple(
            # Never use repr() for `Enum` instances:
            (k, repr(v) if isinstance(v, str) else str(v))
            for k, v in pairs
        )
        
        attrs = ', '.join(('='.join(pair) for pair in stringified))
        s = f"{cls}({attrs})"
        return s

    def __class_getitem__(cls, item: TokenKind) -> str:
        item_name = (
            item.__class__.__name__
            if not hasattr(item, '__name__')
            else item.__name__
        )
        return f"{cls.__name__}[{item_name}]"

    def match_repr(self) -> str | None:
        '''
        Return the token's value in quotes, if parsed as a string,
        else ``None``
        '''
        if not len(self.matchgroups) > 1:
            return None
        quote = self.matchgroups[0]
        val = self.matchgroups[1]
        return f"{quote}{val}{quote}"

    def error_leader(self, with_col: bool = False) -> str:
        '''
        Return the beginning of an error message that features the
        filename, line number and possibly current column number.

        :param with_col: Also print the token's column number,
            defaults to ``False``
        :type with_col: :class:`bool`
        '''
        file = f"File {self.file}, " if self.file is not None else ""
        column = ', column ' + str(self.colno) if with_col else ""
        msg = f"{file}line {self.lineno}{column}: "
        return msg

    def coords(self) -> Location:
        '''
        Return the token's current coordinates as (line, column).
        '''
        return (self.lineno, self.colno)
    
##    def check_expected_after(
##        self,
##        prev: Token,
##        expected: TokenKind | tuple[TokenKind],
##        msg: str = None,
##        error: bool = True
##    ) -> bool | NoReturn:
##        '''
##        '''
##        if isinstance(expected, tuple):
##            invalid = (
##                isinstance(prev, Token)
##                and prev.kind not in expected
##            )
##        else:
##            invalid = (
##                isinstance(expected, Token)
##                and prev.kind is not expected
##            )
##
##        if invalid:
##            if error:
##                if msg is None:
##                    msg = f"Unexpected token: {self.value!r}"
##                raise TokenError.hl_error(self, msg)
##
##            return False
##        return True


class Lexer:
    '''
    The lexer splits text apart into individual tokens.

    :param string: If not using a file, use this string for lexing.
    :type string: :class:`str`

    :param file: The file to use for lexing.
        When unset or ``None``, use `string` by default.
    :type file: :class: `PathLike`
    '''
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
        (re.compile(r'^' + NEWLINE + r'+'), Newline.NEWLINE),
        ## Skip all other whitespace:
        # (re.compile(r'^[^' + NEWLINE + r'\S]+'), Ignore.SPACE),
        (re.compile(r'^[^' + NEWLINE + r'\S]+'), None),

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
        (re.compile(r'^[^' + NOT_IN_IDS + r']+'), Symbol.IDENTIFIER),
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
        self._token_stack = LifoQueue()
        self._tokens = []

        self._cursor = 0  # 0-indexed
        self._lineno = 1  # 1-indexed
        self._colno = 1  # 1-indexed
        self.eof = EndOfFile.EOF

    def lineno(self) -> int:
        '''
        Return the current line number.
        '''
        return self._string[:self._cursor].count('\n') + 1

    def curr_line(self) -> str:
        '''
        Return the text of the current line.
        '''
        return self._lines[self._lineno - 1]

    def get_line(self, lookup: LineNo | Token) -> str:
        '''
        Retrieve a line using its line number or a token.

        :param lookup: Use this line number or token to get the line.
        :type lookup: :class:`LineNo` | :class:`Token`
        :returns: The text of the line gotten using `lookup`
        :rtype: :class:`str`
        '''
        if isinstance(lookup, Token):
            lineno = lookup.at[1][0]
        else:
            lineno = lookup
        return self._lines[lineno - 1]

    def coords(self) -> Location:
        '''
        Return the lexer's current coordinates as (line, column).
        '''
        return (self._lineno, self._colno)
    
    def lex(self, string: str = None) -> list[Token]:
        '''
        Return a list of tokens from lexing.
        Optionally lex a new string `string`.

        :param string: The string to lex, if not `self._string`
        :type string: :class:`str`
        '''
        if string is not None:
            self._string = string

        tokens = []
        try:
            while True:
                tok = self.get_token()
                tokens.append(tok)
                if tok.kind is self.eof:
                    break
        except TokenError as e:
            import traceback
            traceback.print_exc(limit=1)
            raise 

        self.reset()
        return tokens

    def reset(self) -> Self:
        '''
        Move the lexer back to the beginning of the string.
        '''
        self._cursor = 0
        self._lineno = 1
        self._colno = 1
        self._tokens.clear()
        return self

    def get_prev(self, back: int = 1, since: Token = None) -> tuple[Token]:
        '''
        Retrieve tokens from before the current position of the lexer.

        :param back: How many tokens before the current token to look,
            defaults to 1
        :type back: :class:`int`

        :param since: Return every token after this token.
        :type since: :class:`Token`
        '''
        if since is None:
            return self._tokens[-back:]
        tokens = self._tokens
        idx = tuple(tok.cursor for tok in tokens).index(since.cursor)
        ret = tokens[idx - back : idx + 1]
        return ret

    def error_leader(self, with_col: bool = False) -> str:
        '''
        Return the beginning of an error message that features the
        filename, line number and possibly current column number.

        :param with_col: Also print the current column number,
            defaults to ``False``
        :type with_col: :class:`bool`
        '''
        file = self._file if self._file is not None else ''
        column = ', column ' + str(self._colno) if with_col else ''
        msg = f"File {file!r}, line {self._lineno}{column}: "
        return msg

    def get_token(self) -> Token:
        '''
        Return the next token in the lexing stream.

        :raises: :exc:`TokenError` upon an unexpected token
        '''
        # The stack will have contents after string concatenation.
        if not self._token_stack.empty():
            tok = self._token_stack.get()
            self._tokens.append(tok)
            return tok

        # Everything after and including the cursor position:
        s = self._string[self._cursor:]

        # Match against each rule:
        for test, kind in self._rules:
            m = re.match(test, s)

            if m is None:
                # This rule not matched; try the next one.
                continue

            if kind is None:
                # Ignore whitespace and comments:
                l = len(m.group())
                self._cursor += l
                self._colno += l
                return self.get_token()

            tok = Token(
                value=m.group(),
                at=(self._cursor, (self._lineno, self._colno)),
                kind=kind,
                matchgroups=m.groups(),
                lexer=self,
                file=self._file
            )

            # Update location:
            self._cursor += len(tok.value)
            self._colno += len(tok.value)
            if kind is Newline.NEWLINE:
                self._lineno += len(tok.value)
                self._colno = 1

            if kind is Literal.STRING:
                # Process strings by removing quotes:
                speech_char = tok.matchgroups[0]
                tok.value = tok.value.strip(speech_char)

                # Concatenate neighboring strings:
                if self.STRING_CONCAT:
                    while True:
                        maybe_str = self.get_token()
                        if maybe_str.kind in (*Ignore, Newline.NEWLINE):
                            continue
                        break

                    if maybe_str.kind is Literal.STRING:
                        # Concatenate.
                        tok.value += maybe_str.value
                        self._tokens.append(tok)
                        return tok

                    else:
                        # Handle the next token separately.
                        self._token_stack.put(maybe_str)

            self._tokens.append(tok)
            return tok

        else:
            if s is EOF:
                tok = Token(
                    value=s,
                    at=(self._cursor, (self._lineno, self._colno)),
                    kind=self.eof,
                    matchgroups=None,
                    lexer=self,
                    file=self._file
                )

                self._tokens.append(tok)
                return tok

            # If a token is not returned, prepare an error message:
            bad_value = s.split(None, 1)[0]
            bad_token = Token(
                value=bad_value,
                at=(self._cursor, (self._lineno, self._colno)),
                kind=Unknown.UNKNOWN,
                matchgroups=None,
                lexer=self,
                file=self._file
            )
            try:
                if bad_value in self.SPEECH_CHARS:
                    msg = f"Unmatched quote: {bad_value!r}"
                elif bad_value.startswith(self.SPEECH_CHARS):
                    msg = f"Unmatched quote: {bad_value!r}"

                else:
                    msg = f"Unexpected token: {bad_value!r}"

                raise TokenError.hl_error(bad_token, msg)

            except TokenError as e:
                # Avoid recursive stack traces with hundreds of frames:
                import __main__ as possibly_repl
                if not hasattr(possibly_repl, '__file__'):
                    # User is in a REPL! Don't yeet them.
                    raise
                else:
                    # OK, yeet:
                    import traceback
                    traceback.print_exc(limit=1)
                    raise StackTraceSilencer(1)  # Sorry...


class Parser:
    '''
    Parse config files, converting them to abstract syntax trees.

    :param string: If given, parse this string, optional
    :type string: :class:`str`

    :param file: If given, parse this file, optional
    :type file: :class:`PathLike`
    '''
    def __init__(
        self,
        string: str = None,
        *,
        file: PathLike = None,
    ) -> None:
        if string is None:
            if file is None:
                msg = "Either a string or a filename is required"
                raise ValueError(msg)

        self._string = string
        self._file = file
        self._lexer = Lexer(string, file)
        self._tokens = self._lexer.lex()
        self._cursor = 0
        self._current_expr = None

    def tokens(self) -> list[Token]:
        '''
        Return the list of tokens generated by the lexer.
        '''
        return self._lexer.lex()

    def token_dict(self) -> dict[CharNo, Token]:
        '''
        Return a dict mapping lexer cursor value to tokens.
        '''
        return {t.cursor: t for t in self.tokens()}

    def to_dict(self, tree = None) -> dict[str]:
        '''
        Parse a config file and return its data as a :class:`dict`.
        '''
        if tree is None:
            self._lexer.reset()
            tree = self.get_stmts()

        i = Interpreter()
        mapping = {}
        for n in tree:
            if not isinstance(n, Assign):
                # Each statement must map one thing to another.
                # This statement breaks the rules.
                note = (
                    f"{n._token.kind.value} cannot be at"
                    f" the start of a statement"
                )
                msg = f"Invalid syntax: {n._token.match_repr()!r} ({note})"
                raise ParseError.hl_error(n._token, msg)

            parsed = i.visit(n)
            mapping.update(parsed)
        return mapping

    def cur_tok(self) -> Token:
        '''
        Return the current token being parsed.
        '''
        return self._tokens[self._cursor]

    def expect_curr(
        self,
        kind: TokenKind | tuple[TokenKind],
        errmsg: str = None
    ) -> NoReturn | bool:
        '''
        Test if the current token matches a certain kind given by `kind`.
        If the test fails, raise :exc:`ParseError` by default using
        `errmsg` or return ``False`` if `errmsg` is ``None``.
        If the test passes, return the current token.

        :param kind: The kind(s) to expect from the current token
        :type kind: :class:`TokenKind`

        :param errmsg: The error message to display, optional
        :type errmsg: :class:`str`
        '''
        if not isinstance(kind, tuple):
            kind = (kind,)
        tok = self.cur_tok()
        if tok.kind not in kind:
            if errmsg is None:
                return False
            raise ParseError.hl_error(tok, errmsg)
        return tok

    def expect_next(
        self,
        kind: TokenKind | tuple[TokenKind],
        errmsg: str
    ) -> NoReturn | bool:
        '''
        Test if the next token matches a certain kind given by `kind`.
        If the test fails, raise :exc:`ParseError` by default using
        `errmsg` or return ``False`` if `errmsg` is ``None``.
        If the test passes, return the current token.

        :param kind: The kind(s) to expect from the current token
        :type kind: :class:`TokenKind`

        :param errmsg: The error message to display, optional
        :type errmsg: :class:`str`
        '''
        if not isinstance(kind, tuple):
            kind = (kind,)
        tok = self.advance()
        if tok.kind not in kind:
            if errmsg is None:
                return False
            raise ParseError.hl_error(tok, errmsg)
        return tok

    def advance(self) -> Token:
        '''
        Move to the next token. Return that token.
        '''
        if self._cursor < len(self._tokens) - 1:
            self._cursor += 1
        curr = self.cur_tok()
        return curr

    def to_next(self) -> Token:
        '''
        Advance to the next non-whitespace token. Return that token.
        '''
        while True:
            if (tok := self.advance()).kind in (*Ignore, Newline.NEWLINE):
                self.advance()
            return tok

    def reset(self) -> None:
        '''
        Bring the lexer back to the first token.
        '''
        self._cursor = 0

    def not_eof(self) -> bool:
        '''
        Return whether the current token is the end-of-file.
        '''
        return (self.cur_tok().kind is not EndOfFile.EOF)

    def to_skip(self) -> bool:
        '''
        Return whether the current token is whitespace or a comment.
        '''
        return (self.cur_tok().kind in (*Ignore, Newline.NEWLINE))

    def parse_root_assign(self) -> Assign:
        '''
        Parse an assignment in the outermost scope at the current token::

            foo = 'bar'
        '''
        msg = "Invalid syntax (expected an identifier):"
        target_tok = self.expect_curr(Symbol.IDENTIFIER, msg)
        self._current_expr = Assign
        target = Name(target_tok.value)
        target._token = target_tok

        # Ignore newlines between identifier and operator:
        self.to_next()

        possible_starts = (BinOp.ASSIGN, Syntax.L_CURLY_BRACE)
        msg = "Invalid syntax (expected '=' or '{'):"
        operator = self.expect_curr(possible_starts, msg)
        if operator.kind is Syntax.L_CURLY_BRACE:
            value = self.parse_object()
        else:
            value = self.parse_expr()
        node = Assign([target], value)
        node._token = operator

        # Advance to the next assignment:
        self.to_next()
        return node

    def parse_expr(self) -> AST:
        '''
        Parse an expression at the current token.
        '''
        while True:
            tok = self.advance()
            kind = tok.kind

            match kind:
                case EndOfFile.EOF:
                    return kind
                case kind if kind in Ignore:
                    continue
                case Newline.NEWLINE | Syntax.COMMA:
                    if self._current_expr in (Assign, Dict):
                        node = Constant(None)
                    else:
                        continue
                case kind if kind in Literal:
                    node = self.parse_literal(tok)
                case BinOp.ATTRIBUTE:
                    node = self.parse_mapping(self.parse_attribute(tok))
                case Syntax.L_BRACKET:
                    node = self.parse_list()
                case Syntax.L_CURLY_BRACE:
                    node = self.parse_object()
                case Symbol.IDENTIFIER:
                    node = Name(tok.value)
                case _:
                    return tok

            node._token = tok
            return node

    def parse_list(self) -> List:
        '''
        Parse a list at the current token.
        '''
        msg = "parse_list() called at the wrong time"
        self.expect_curr(Syntax.L_BRACKET, msg)
        self._current_expr = List
        elems = []
        while True:
            elem = self.parse_expr()
            if isinstance(elem, Token):
                if elem.kind is Syntax.R_BRACKET:
                    break
            elems.append(elem)
        return List(elems)
        
    def parse_mapping(self) -> Dict:
        '''
        Parse a 1-to-1 mapping at the current token inside an object::

            {key = 'val'}
        '''
        msg = (
            "Invalid syntax (expected an identifier):",
            "Invalid syntax (expected '=' or '.'):"
        )
        self._current_expr = Dict
        target_tok = self.expect_curr(Symbol.IDENTIFIER, msg[0])
        target = Name(target_tok.value)
        target._token = target_tok

        maybe_attr = self.advance()
        if maybe_attr.kind is BinOp.ATTRIBUTE:
            target = self.parse_attribute(target)

        assign_tok = self.expect_curr(BinOp.ASSIGN, msg[1])
        value = self.parse_expr()
        # Disallow assigning identifiers:
        if isinstance(value, Name):
            typ = value._token.kind.value
            val = repr(value._token.value)
            note = f"expected expression, got {typ} {val}"
            msg = f"Invalid assignment: {note}:"
            raise ParseError.hl_error(value._token, msg)
        node = Dict([target], [value])
        node._token = assign_tok
        return node

    def parse_attribute(self, outer: Name | Attribute) -> Attribute:
        '''
        Parse an attribute at the current token inside an object::

            a.b.c
        '''
        msg = (
            "parse_attribute() called at the wrong time",
            "Invalid syntax (expected an identifier):",
        )
        operator = self.expect_curr(BinOp.ATTRIBUTE, msg[0])
        maybe_base = self.expect_next(Symbol.IDENTIFIER, msg[1])
        attr = maybe_base.value
        target = Attribute(outer, attr)
        maybe_another = self.advance()
        target._token = maybe_base
        if maybe_another.kind is BinOp.ATTRIBUTE:
            target = self.parse_attribute(target)
        return target

    def parse_object(self) -> Dict:
        '''
        Parse a dict containing many mappings at the current token::

            {
                foo = 'bar'
                baz = 42
                ...
            }

        '''
        msg = (
            "parse_object() called at the wrong time",
        )
        self.expect_curr(Syntax.L_CURLY_BRACE, msg[0])
        self._current_expr = Dict
        keys = []
        vals = []

        while True:
            tok = self.advance()
            kind = tok.kind
            match kind:
                case Syntax.R_CURLY_BRACE:
                    break
                case kind if kind in (*Ignore, Syntax.COMMA, Newline.NEWLINE):
                    continue
                case Symbol.IDENTIFIER:
                    pair = self.parse_mapping()
                    key = pair.keys[-1]  # Only one key and one value
                    val = pair.values[-1]
                    if key not in keys:
                        keys.append(key)
                        vals.append(val)
                    else:
                        # Make nested dicts.
                        idx = keys.index(key)
                        # Equivalent to dict.update():
                        vals[idx].keys = val.keys
                        vals[idx].values = val.values
                case _:
                    typ = kind.value
                    val = repr(tok.value)
                    note = f"expected variable name, got {typ} {val}"
                    msg = f"Invalid target for assignment: {note}:"
                    raise ParseError.hl_error(tok, msg)

        # Put all the assignments together:
        return Dict(keys, vals)

    def parse_literal(self, tok) -> Constant:
        '''
        Parse a literal constant value at the current token::

            42
        '''
        self._current_expr = Constant
        match tok.kind:
            # Return literals as Constants:
            case Literal.STRING:
                value = tok.value
            case Literal.INTEGER:
                value = int(tok.value)
            case Literal.FLOAT:
                value = float(tok.value)
            case Literal.TRUE:
                value = True
            case Literal.FALSE:
                value = False
            case _:
                raise ParseError("Expected a literal, but got" + repr(tok))
        return Constant(value)

    def parse_stmt(self) -> Assign:
        '''
        Parse an assignment statement at the current token.
        '''
        while self.not_eof():
            tok = self.cur_tok()
            kind = tok.kind
            assign = self.parse_root_assign()
            return assign

    def get_stmts(self) -> list[Assign]:
        '''
        Parse many assignment statements and return them in a list.
        '''
        self.reset()
        assignments = []
        while self.not_eof():
            assignments.append(self.parse_stmt())
        self.reset()
        return assignments

    def parse(self) -> Module:
        '''
        Parse the lexer stream and return it as a :class:`Module`.
        '''
        return Module(self.get_stmts())


class ConfigDictUnparser:
    '''
    Convert Python dicts to ASTs.
    '''
    @classmethod
    def unparse(cls, mapping: Mapping) -> Module:
        '''
        Convert a dict to a Module AST.

        :param mapping: The dict to convert
        :type mapping: :class:`Mapping`
        '''
        if not isinstance(mapping, Mapping):
            return mapping
        assignments = []
        for key, val in mapping.items():
            node = Assign([Name(key)], cls.unparse_node(val))
            assignments.append(node)
        return Module(assignments)


    @classmethod
    def _unparse_dict(
        cls,
        dct: Mapping | Any,
        attrs: list[str] = [],
        # root_name: str = None,
    ) -> tuple[list[AST]]:
        '''
        Unparse a nested dictionary.

        :param dct: The dict to unparse
        :type dct: :class:`Mapping` | :class:`Any`

        :param attrs: The list of dict keys that have been seen
        :type attrs: :class:`MutableSequence`

        :param root_name: The uppermost dict key
        :type root_name: :class:`str`
        '''

        print(f"Processing {dct = }, {attrs = }")
        if not isinstance(dct, Mapping):
            return attrs, dct

        keys = []
        vals = []
        for key, val in dct.items():
            print(key, val)

            if not attrs:
                # The root attribute
                attrs = [key]
                attrs, assign = cls._unparse_dict(val, attrs)

            elif isinstance(val, Mapping):
                # A nested attrubute
                attrs.append(key)
                attrs, assign = cls._unparse_dict(val, attrs)
                vals.append(assign)
            else:
                attrs.append(key)
                vals.append(val)
            print((attrs, vals))

        return attrs, vals

    @classmethod
    def _dict_to_list(cls, dct, seen=[]) -> tuple[list[list], list]:
        # {'a': {'b': {'c': assign}, {'bb': assign}}} -> [[a, b, c], [a, bb]]
        # seen = []
        nodes = []
        vals = []
        print(f"{dct = }, {seen = }")
        if not isinstance(dct, Mapping):
            return [seen], [dct]
        if len(dct) == 1:
            print("Going the short way")
            for a, v in dct.items():
                # print(f"{a = }, {v = }, {oldseen = }")
                print(f"{a = }")
                seen.append(a)
                return cls._dict_to_list(v, seen)
                return [[a]], v
        oldseen = seen
        for a, v in dct.items():
            print(f"{a = }, {v = }, {nodes = }")
            seen.append(a)
            # 'a', {...}'
            if not isinstance(v, Mapping):
##                # {'b': {'c': assign}}
##                print("v is a mapping")
##                attrs, values = cls._dict_to_list(v, oldseen)
##                seen.append(attrs)
##                nodes.extend(attrs)
##                vals.extend(values)
##            else:
                print(f"{seen = }")
                # print(f"{oldseen + seen = }")
                nodes.append(seen)
                vals.append(v)
            # seen = oldseen[:-2]
            seen = seen[:-1]
        print(f"{nodes = }")
        # print(f"{nodes = }, {vals = }")
        return nodes, vals
        return seen, vals



    @classmethod
    def _list_to_attributes(cls, attrs) -> list[Attribute]:
        nodes = []
        spent = []
        if not isinstance(attrs, Sequence):
            return attrs
            # return [attrs]
            # return attrs
            # return node
        started = False
        for agroup in attrs:
            node = Name(agroup.pop(0))
            # node = Name(agroup[0])
            for a in agroup:
                node = Attribute(node, a)
            nodes.append(node)

            # node = Attribute(node, cls._list_to_attributes(attr))
            # nodes.append(node)
        return nodes


    @classmethod
    def unparse_node(cls, thing) -> AST:
        '''
        '''
        if isinstance(thing, Mapping):
            keys = []
            vals = []
            for key, val in thing.items():
                print(f"{key = }, {val = }")
                # print(key, val)

                if isinstance(val, Mapping):
                    # An attribute, possibly nested.
                    # attr, assign = cls._unparse_dict(val)
                    attrs, assigns = cls._dict_to_list(val, [key])
                    print(attrs, assigns)
                    nodes = cls._list_to_attributes(attrs) 
                    # node = Attribute(Name(key), attr)
                    # attr.insert(0, key)
                    # node = cls._list_to_attributes(attr)
                    # v = cls.unparse_node(assign)
                    keys.extend(nodes)
                    # vals.extend(cls.unparse_node(v) for v in assigns)
                    vals.extend(assigns)
                else:
                    # An explicit assignment.
                    node = Name(key)
                    v = cls.unparse_node(val)
                ### NOOO:
                # Dicts can't contain duplicates, so it's safe to append:
##                if key not in key_ref:
##                    key_ref.append(key)
                # print(v)
                    keys.append(node)
                    vals.append(v)
##                else:
##                    print("HOW?")
##                    idx = key_ref.index(key)
##                    vals[idx].append(v)

                # print(key_ref)
                # print(f"{keys = }, {vals = }")
            return Dict(keys, [cls.unparse_node(v) for v in vals])

        elif isinstance(thing, list):
            values = [Name(v) for v in thing]
            return List(values)

        elif isinstance(thing, (int, float, str)):
            return Constant(thing)

        elif thing in (None, True, False):
            return Constant(thing)

        else:
            return thing

    @classmethod
    def _nested_update(
        cls,
        dct: MutableMapping | Any,
        upd: Mapping,
        assign: Any
    ) -> dict:
        if not isinstance(dct, MutableMapping):
            return {upd.popitem()[0]: assign}
        for k, v in upd.items():
            if v is self._EXPR_PLACEHOLDER:
                v = assign
            if isinstance(v, Mapping):
                dct[k] = cls._nested_update(dct.get(k, {}), v, assign)
            else:
                dct[k] = v
        return dct

    @classmethod
    def to_file(cls, mapping: Mapping) -> FileContents:
        '''
        '''
        tree = cls.unparse(mapping)
        text = ConfigFileMaker().stringify(tree.body)
        return text


class Interpreter(NodeVisitor):
    '''
    Convert ASTs to literal Python code.
    '''
    _EXPR_PLACEHOLDER = '_EXPR_PLACEHOLDER'

    def visit_Assign(self, node: Assign) -> dict:
        return {self.visit((node.targets[-1])): self.visit(node.value)}

    def visit_Attribute(self, node: Attribute) -> dict:
        attrs = self._process_nested_attr(node, [])
        new_d = self._EXPR_PLACEHOLDER
        for attr in attrs:
            new_d = {attr: new_d}
        return new_d

    def _process_nested_attr(
        self,
        node: Attribute | Name | str,
        attrs: list[str]
    ) -> dict:
        '''
        '''
        if not attrs:
            attrs = [node.attr]
            return self._process_nested_attr(node.value, attrs)
        if isinstance(node, Attribute):
            attrs.append(node.attr)
            return self._process_nested_attr(node.value, attrs)
        if isinstance(node, Name):
            return self._process_nested_attr(node.id, attrs)
        if isinstance(node, str):
            attrs.append(node)
            return attrs

    def visit_Constant(self, node: Constant) -> str | int | float | None:
        return node.value

    def _nested_update(
        self,
        orig: Mapping | Any,
        upd: Mapping,
        assign: Any
    ) -> dict:
        upd = upd.copy()
        if not isinstance(orig, Mapping):
            return self._nested_update({}, upd, assign)
        orig = orig.copy()
        for k, v in upd.items():
            if v is self._EXPR_PLACEHOLDER:
                v = assign
            if isinstance(v, Mapping):
                updated = self._nested_update(orig.get(k, {}), v, assign)
                orig[k] = updated
            else:
                orig[k] = v
        return orig

    def visit_Dict(self, node: Dict) -> dict[str]:
        new_d = {}
        for key, val in zip(node.keys, node.values):
            target = self.visit(key)
            value = self.visit(val)
            if isinstance(target, Mapping):
                # An Attribute
                new_d = self._nested_update(new_d, target, value)
            else:
                # An assignment
                new_d.update({target: value})
        return new_d

    def visit_List(self, node: List) -> list:
        return [self.visit(e) for e in node.elts]

    def visit_Name(self, node: Name) -> str:
        return node.id


class ConfigFileMaker(NodeVisitor):
    '''
    Convert ASTs to string representations with config file syntax.
    '''
    def __init__(self) -> None:
        self.indent = 4 * ' '

    def stringify(self, tree: Module, sep: str = '\n\n') -> FileContents:
        strings = [self.visit(n) for n in tree]
        return sep.join(strings)

    def visit_Attribute(self, node: Attribute) -> str:
        base = self.visit(node.value)
        attr = self.visit(node.attr)
        return f"{base}.{attr}"

    def visit_Assign(self, node: Assign) -> str:
        target = self.visit(node.targets[-1])
        value = self.visit(node.value)
        return f"{target} = {value}"

    def visit_Constant(self, node: Constant) -> str:
        val = node.value
        if isinstance(val, str):
            return repr(val)
        if val is None:
            return ""
        return str(val)

    def visit_Dict(self, node: Dict) -> str:
        keys = (self.visit(k) for k in node.keys)
        values = (self.visit(v) for v in node.values)
        assignments = (' = '.join(pair) for pair in zip(keys, values))
        joined = f"\n{self.indent}".join(assignments)
        string = f"{{\n{self.indent}{joined}\n}}"
        return string

    def visit_List(self, node: List) -> str:
        one_line_limit = 3
        elems = tuple(self.visit(e) for e in node.elts)
        if len(elems) > one_line_limit:
            joined = ('\n' + self.indent).join(elems)
            string = f"[\n{self.indent}{joined}\n]"
        else:
            joined = ', '.join(elems)
            string = f"[{joined}]"
        return string

    def visit_Name(self, node: Name) -> str:
        string = node.id
        return string


if __name__ == '__main__':
##    print("AST parsed from file contents:")
    p = Parser(file=FILE)
##    print(ast.dump(p.parse()))
##    print()
##
##    print("Dict built from AST parsed from file contents:")
    d = p.to_dict()
    print(d)
##    print()
##
##    print("AST from unparsing dict built from AST parsed from file contents:")
    u = ConfigDictUnparser.unparse(d)
    # u = p.parse()
    print(ast.dump(u.body[-1].value))
    print()

##    print("File contents built from AST from unparsing dict built from AST parsed from file contents:")
##    c = ConfigFileMaker().stringify(u.body)
##    c = ConfigDictUnparser.to_file(d)
##    # c  = u.as_file()
##    print("```")
##    print(c)
##    print("```")
##    print()

