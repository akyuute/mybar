__all__ = (
    'Token',
    'Lexer',
    'FileParser',
    'DictParser',
    'Unparser',
    'ConfigFileMaker',
)


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

from ._types import FileContents

Token = TypeVar('Token')
Lexer = TypeVar('Lexer')


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]


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
    

def unescape_backslash(s: str, encoding: str = 'utf-8') -> str:
    '''
    Unescape characters escaped by backslashes.

    :param s: The string to escape
    :type s: :class:`str`

    :param encoding: The encoding `s` comes in, defaults to ``'utf-8'``
    :type encoding: :class:`str`
    '''
    return (
        s.encode(encoding)
        .decode('unicode-escape')
        # .encode(encoding)
        # .decode(encoding)
    )


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
                value = tok.value.strip(speech_char)
                if '\\' in value:
                    value = unescape_backslash(value)
                tok.value = value

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


class FileParser:
    '''
    Parse config files, converting them to abstract syntax trees.

    :param file: If given, parse this file, optional
    :type file: :class:`PathLike`

    :param string: If given, parse this string, optional
    :type string: :class:`str`
    '''
    def __init__(
        self,
        file: PathLike = None,
        *,
        string: str = None,
    ) -> None:
        if string is None:
            if file is None:
                msg = "Either a string or a filename is required"
                raise ValueError(msg)

        self._file = file
        self._string = string
        self._lexer = Lexer(string, file)
        self._tokens = self._lexer.lex()
        self._cursor = 0
        self._current_expr = None

    def tokens(self) -> list[Token]:
        '''
        Return the list of tokens generated by the lexer.
        '''
        return self._lexer.lex()

    def _token_dict(self) -> dict[CharNo, Token]:
        '''
        Return a dict mapping lexer cursor value to tokens.
        '''
        return {t.cursor: t for t in self.tokens()}

    def _cur_tok(self) -> Token:
        '''
        Return the current token being parsed.
        '''
        return self._tokens[self._cursor]

    def _advance(self) -> Token:
        '''
        Move to the next token. Return that token.
        '''
        if self._cursor < len(self._tokens) - 1:
            self._cursor += 1
        curr = self._cur_tok()
        return curr

    def _next(self) -> Token:
        '''
        Advance to the next non-whitespace token. Return that token.
        '''
        while True:
            if (tok := self._advance()).kind in (*Ignore, Newline.NEWLINE):
                self._advance()
            return tok

    def _expect_curr(
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
        tok = self._cur_tok()
        if tok.kind not in kind:
            if errmsg is None:
                return False
            raise ParseError.hl_error(tok, errmsg)
        return tok

    def _expect_next(
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
        tok = self._next()
        if tok.kind not in kind:
            if errmsg is None:
                return False
            raise ParseError.hl_error(tok, errmsg)
        return tok

    def _reset(self) -> None:
        '''
        Bring the lexer back to the first token.
        '''
        self._cursor = 0

    def _not_eof(self) -> bool:
        '''
        Return whether the current token is the end-of-file.
        '''
        return (self._cur_tok().kind is not EndOfFile.EOF)

    def _should_skip(self) -> bool:
        '''
        Return whether the current token is whitespace or a comment.
        '''
        return (self._cur_tok().kind in (*Ignore, Newline.NEWLINE))

    def _parse_root_assign(self) -> Assign:
        '''
        Parse an assignment in the outermost scope at the current token::

            foo = 'bar'
        '''
        msg = "Invalid syntax (expected an identifier):"
        target_tok = self._expect_curr(Symbol.IDENTIFIER, msg)
        self._current_expr = Assign
        target = Name(target_tok.value)
        target._token = target_tok

        # Ignore newlines between identifier and operator:
        self._next()

        possible_starts = (BinOp.ASSIGN, Syntax.L_CURLY_BRACE)
        msg = "Invalid syntax (expected '=' or '{'):"
        operator = self._expect_curr(possible_starts, msg)
        if operator.kind is Syntax.L_CURLY_BRACE:
            value = self._parse_object()
        else:
            value = self._parse_expr()
        node = Assign([target], value)
        node._token = operator

        # Advance to the next assignment:
        self._next()
        return node

    def _parse_expr(self) -> AST:
        '''
        Parse an expression at the current token.
        '''
        while True:
            tok = self._advance()
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
                    node = self._parse_literal(tok)
                case BinOp.ATTRIBUTE:
                    node = self._parse_mapping(self._parse_attribute(tok))
                case Syntax.L_BRACKET:
                    node = self._parse_list()
                case Syntax.L_CURLY_BRACE:
                    node = self._parse_object()
                case Symbol.IDENTIFIER:
                    node = Name(tok.value)
                case _:
                    return tok

            node._token = tok
            return node

    def _parse_list(self) -> List:
        '''
        Parse a list at the current token.
        '''
        msg = "_parse_list() called at the wrong time"
        self._expect_curr(Syntax.L_BRACKET, msg)
        self._current_expr = List
        elems = []
        while True:
            elem = self._parse_expr()
            if isinstance(elem, Token):
                if elem.kind is Syntax.R_BRACKET:
                    break
            elems.append(elem)
        return List(elems)
        
    def _parse_mapping(self) -> Dict:
        '''
        Parse a 1-to-1 mapping at the current token inside an object::

            {key = 'val'}
        '''
        msg = (
            "Invalid syntax (expected an identifier):",
            "Invalid syntax (expected '=' or '.'):"
        )
        self._current_expr = Dict
        target_tok = self._expect_curr(Symbol.IDENTIFIER, msg[0])
        target = Name(target_tok.value)
        target._token = target_tok

        maybe_attr = self._next()
        if maybe_attr.kind is BinOp.ATTRIBUTE:
            target = self._parse_attribute(target)

        assign_tok = self._expect_curr(BinOp.ASSIGN, msg[1])
        value = self._parse_expr()
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

    def _parse_attribute(self, outer: Name | Attribute) -> Attribute:
        '''
        Parse an attribute at the current token inside an object::

            a.b.c
        '''
        msg = (
            "_parse_attribute() called at the wrong time",
            "Invalid syntax (expected an identifier):",
        )
        operator = self._expect_curr(BinOp.ATTRIBUTE, msg[0])
        maybe_base = self._expect_next(Symbol.IDENTIFIER, msg[1])
        attr = maybe_base.value
        target = Attribute(outer, attr)
        maybe_another = self._next()
        target._token = maybe_base
        if maybe_another.kind is BinOp.ATTRIBUTE:
            target = self._parse_attribute(target)
        return target

    def _parse_object(self) -> Dict:
        '''
        Parse a dict containing many mappings at the current token::

            {
                foo = 'bar'
                baz = 42
                ...
            }

        '''
        msg = (
            "_parse_object() called at the wrong time",
        )
        self._expect_curr(Syntax.L_CURLY_BRACE, msg[0])
        self._current_expr = Dict
        keys = []
        vals = []

        while True:
            tok = self._next()
            kind = tok.kind
            match kind:
                case Syntax.R_CURLY_BRACE:
                    break
                case kind if kind in (*Ignore, Syntax.COMMA, Newline.NEWLINE):
                    continue
                case Symbol.IDENTIFIER:
                    pair = self._parse_mapping()
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

    def _parse_literal(self, tok) -> Constant:
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

    def _parse_stmt(self) -> Assign:
        '''
        Parse an assignment statement at the current token.
        '''
        while self._not_eof():
            tok = self._cur_tok()
            kind = tok.kind
            assign = self._parse_root_assign()
            return assign

    def _get_stmts(self) -> list[Assign]:
        '''
        Parse many assignment statements and return them in a list.
        '''
        self._reset()
        assignments = []
        while self._not_eof():
            assignments.append(self._parse_stmt())
        self._reset()
        return assignments

    def parse(self) -> Module:
        '''
        Parse the lexer stream and return it as a :class:`Module`.
        '''
        return Module(self._get_stmts())

    def to_dict(self, tree: AST = None) -> dict[str]:
        '''
        Parse a config file and return its data as a :class:`dict`.
        '''
        if tree is None:
            self._lexer.reset()
            tree = self._get_stmts()

        u = Unparser()
        mapping = {}
        for node in tree:
            if not isinstance(node, Assign):
                # Each statement must map one thing to another.
                # This statement breaks that rule.
                note = (
                    f"{node._token.kind.value} cannot be at"
                    f" the start of a statement"
                )
                msg = f"Invalid syntax: {node._token.match_repr()!r} ({note})"
                raise ParseError.hl_error(node._token, msg)

            unparsed = u.visit(node)
            mapping.update(unparsed)
        return mapping


class DictParser:
    '''
    Convert Python dicts to ASTs.
    '''
    @classmethod
    def _process_nested_dict(
        cls,
        dct: Mapping | Any,
        roots: Sequence[str] = [],
        descended: int = 0
    ) -> tuple[list[list[str]], list[object]]:
        nodes = []
        vals = []
        if not isinstance(dct, Mapping):
            # An assignment.
            return ([roots], [dct])

        if len(dct) == 1:
            # An attribute.
            for a, v in dct.items():
                roots.append(a)
                return cls._process_nested_dict(v, roots, descended)

        descended = 0  # Start of a tree.
        for attr, v in dct.items():
            roots.append(attr)
            if isinstance(v, Mapping):
                if descended < len(roots):
                    descended = -len(roots)
                # Descend into lower tree.
                inner_nodes, inner_vals = cls._process_nested_dict(
                    v,
                    roots,
                    descended
                )
                nodes.extend(inner_nodes)
                vals.extend(inner_vals)
            else:
                nodes.append(roots)
                vals.append(v)
            roots = roots[:-descended - 1]  # Reached end of tree.
        return nodes, vals

    @classmethod
    def _process_nested_attrs(
        cls,
        attrs: Sequence[Sequence[str]]
    ) -> list[Attribute]:
        '''
        '''
        nodes = []
        spent = []
        if not isinstance(attrs, Sequence):
            return attrs
        for node_attrs in attrs:
            node_attrs = node_attrs[:]
            node = Name(node_attrs.pop(0))
            for attr in node_attrs:
                node = Attribute(node, attr)
            nodes.append(node)
        return nodes

    @classmethod
    def _parse_node(cls, thing) -> AST:
        '''
        '''
        if isinstance(thing, Mapping):
            keys = []
            vals = []
            for key, val in thing.items():

                if isinstance(val, Mapping):
                    # An attribute, possibly nested.
                    attrs, assigns = cls._process_nested_dict(val, [key])
                    nodes = cls._process_nested_attrs(attrs)
                    keys.extend(nodes)
                    vals.extend(assigns)
                else:
                    # An explicit assignment.
                    node = Name(key)
                    keys.append(node)
                    v = cls._parse_node(val)
                    vals.append(v)
            return Dict(keys, [cls._parse_node(v) for v in vals])

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
    def parse(cls, mapping: Mapping) -> Module:
        '''
        Convert a dict to a Module AST.

        :param mapping: The dict to convert
        :type mapping: :class:`Mapping`
        '''
        if not isinstance(mapping, Mapping):
            return mapping
        assignments = []
        for key, val in mapping.items():
            node = Assign([Name(key)], cls._parse_node(val))
            assignments.append(node)
        return Module(assignments)

    @classmethod
    def make_file(cls, mapping: Mapping) -> FileContents:
        '''
        '''
        tree = cls.parse(mapping)
        text = ConfigFileMaker().stringify(tree.body)
        return text


class Unparser(NodeVisitor):
    '''
    Convert ASTs to Python data structures.
    '''
    _EXPR_PLACEHOLDER = '_EXPR_PLACEHOLDER'

    def unparse(self, node: AST) -> dict:
        return self.visit(node)

    def visit_Assign(self, node: Assign) -> dict:
        return {self.visit((node.targets[-1])): self.visit(node.value)}

    def visit_Attribute(self, node: Attribute) -> dict:
        attrs = self._nested_attr_to_dict(node)
        new_d = self._EXPR_PLACEHOLDER
        for attr in attrs:
            new_d = {attr: new_d}
        return new_d

    def _nested_attr_to_dict(
        self,
        node: Attribute | Name,
        attrs: Sequence[str] = []
    ) -> dict | list[str]:
        '''
        '''
        if isinstance(node, Name):
            attrs.append(node.id)
            return attrs
        if not attrs:
            # A new nested Attribute.
            attrs = [node.attr]
        elif isinstance(node, Attribute):
            attrs.append(node.attr)
        n = node.value
        return self._nested_attr_to_dict(n, attrs)

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

    def visit_Module(self, node: Module) -> list[dict[str]]:
        return [self.visit(n) for n in node.body]

    def visit_Name(self, node: Name) -> str:
        return node.id


class ConfigFileMaker(NodeVisitor):
    '''
    Convert ASTs to string representations with config file syntax.
    '''
    def __init__(self) -> None:
        self.indent = 4 * ' '

    def stringify(
        self,
        tree: Sequence[AST] | Module,
        sep: str = '\n\n'
    ) -> FileContents:
        '''
        '''
        if isinstance(tree, Module):
            tree = tree.body
        strings = [self.visit(n) for n in tree]
        return sep.join(strings)

    def visit_Attribute(self, node: Attribute) -> str:
        base = self.visit(node.value)
        attr = node.attr
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


parsed = FileParser('/home/sam/.config/mybar/mybar.conf').parse()
print(parsed)
print(Unparser().unparse(parsed))
def parse(file: PathLike = None, string: str = None) -> Module:
    return FileParser(file, string).parse()

def unparse(node: AST) -> str:
    pass

