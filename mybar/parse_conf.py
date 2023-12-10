##__all__ = (
##    'ast_to_data',
##    'convert_file',
##    'data_to_text',
##    'parse',
##    'text_to_data',
##    'Grammar',
##)


import ast
import re
import os
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
from itertools import zip_longest
from os import PathLike
from queue import LifoQueue
from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
    MutableSequence
)
from types import GeneratorType
from typing import (
    Any,
    Iterable,
    Literal,
    NamedTuple,
    NoReturn,
    Self,
    TypeAlias,
    TypeVar
)

from ._types import FileContents, PythonData
from .constants import CONFIG_FILE

Token = TypeVar('Token')
Lexer = TypeVar('Lexer')


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]
ConstantValue: TypeAlias = str | int | float | bool | Literal[None]


EOF = ''
NEWLINE = '\n'
NOT_IN_IDS = string.punctuation.replace('_', r'\s')
KEYWORDS = ()


class Grammar:
    '''
.. code::

    Module : Assignment
           | Assignment Delimiter Module

    Assignment : Identifier MaybeEQ Expr

    Identifier : Dotted Identifier
               | IDENTIFIER

    Dotted : IDENTIFIER ATTRIBUTE

    MaybeEQ : EQUALS | None

    Expr : LITERAL
         | List
         | Dict
         | Delimiter
         | None

    List : L_BRACKET RepeatedExpr R_BRACKET

    RepeatedExpr : Expr Delimiter RepeatedExpr
                 | None

    Dict : L_CURLY_BRACE RepeatedKVP R_CURLY_BRACE

    RepeatedKVP : KVPair Delimiter RepeatedKVP
                | None

    KVPair : Identifier MaybeEQ Expr

    Delimiter : NEWLINE | COMMA | None
    '''


class TokKind(Enum):
    NEWLINE = repr(NEWLINE)
    EOF = repr(EOF)

    # Ignored tokens:
    COMMENT = 'COMMENT'
    SPACE = 'SPACE'

    # Literals:
    FLOAT = 'FLOAT'
    INTEGER = 'INTEGER'
    STRING = 'STRING'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    NONE = 'NONE'

    # Symbols:
    IDENTIFIER = 'IDENTIFIER'
    KEYWORD = 'KEYWORD'

    # Binary operators:
    EQUALS = '='
    ATTRIBUTE = '.'
    ADD = '+'
    SUBTRACT = '-'

    # Syntax:
    COMMA = ','
    L_PAREN = '('
    R_PAREN = ')'
    L_BRACKET = '['
    R_BRACKET = ']'
    L_CURLY_BRACE = '{'
    R_CURLY_BRACE = '}'

    UNKNOWN = 'UNKNOWN'

    def __or__(self, other: Any) -> str:
        return type(self).__name__ + ' | ' + type(other).__name__


# Token groups
T_Literal = {
    TokKind.STRING,
    TokKind.INTEGER,
    TokKind.FLOAT,
    TokKind.TRUE,
    TokKind.FALSE,
    TokKind.NONE,
}
T_AssignEvalNone = {TokKind.NEWLINE, TokKind.EOF}
T_Ignore = {TokKind.SPACE, TokKind.COMMENT}
T_Invisible = {*T_Ignore, TokKind.NEWLINE}
T_Syntax = {
    TokKind.L_BRACKET,
    TokKind.L_CURLY_BRACE,
    TokKind.L_PAREN,
    TokKind.R_BRACKET,
    TokKind.R_CURLY_BRACE,
    TokKind.R_PAREN,
}


# FIRST() functions
FIRST: TypeAlias = set
MatchesDotted: FIRST = {TokKind.IDENTIFIER}
MatchesMaybeEQ: FIRST = {TokKind.EQUALS, None}
MatchesExpr: FIRST = {
    *T_Literal,
    TokKind.L_BRACKET,
    TokKind.L_CURLY_BRACE,
    TokKind.NEWLINE,
    TokKind.COMMA,
    None
}
MatchesList: FIRST = {TokKind.L_BRACKET}
MatchesRepeatedExpr: FIRST = {*MatchesExpr, None}
MatchesDict: FIRST = {TokKind.L_CURLY_BRACE}
MatchesRepeatedKVP: FIRST = {TokKind.IDENTIFIER, None}
MatchesKVPair: FIRST = {TokKind.IDENTIFIER}
MatchesDelimiter: FIRST = {TokKind.NEWLINE, TokKind.COMMA, None}

# Return types
Expr: TypeAlias = Constant | List | Dict | Name
Identifier: TypeAlias = Name | Attribute
Delimiter: TypeAlias = None
Dotted: TypeAlias = tuple[Name, bool]
MaybeEQ: TypeAlias = None
RepeatedExpr: TypeAlias = list[Expr]
KVPair: TypeAlias = tuple[Identifier, Expr]
RepeatedKVP: TypeAlias = list[KVPair]


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
            defaults to that of the first token
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
        if leader is None:
            leader = first_tok.error_leader()

        if indent is None:
            indent = 0
        dent = ' ' * indent

        max_len = 100
        break_line = "\n" + dent if len(leader + msg) > max_len else ""
        dent = 2 * dent  # Double indent for following lines.
        highlight = ""

        if len(tokens) == 1:
            line_bridge = " "
            line = lexer.get_line(first_tok)
            if first_tok.kind is TokKind.STRING:
                text = first_tok.match_repr
            else:
                text = first_tok.value
            between = tokens

        else:
            # Highlight multiple tokens using all in the range:
            line_bridge = " "
            between = lexer.in_range(tokens[0], tokens[-1])
            if not between:
                between = tokens
            if any(t.kind is TokKind.NEWLINE for t in between):
                # Consolidate multiple lines:
                with_dups = (
                    lexer.get_line(t) for t in between
                    if t.kind not in T_Ignore
                )
                lines = dict.fromkeys(with_dups)
                # Don't count line breaks twice:
                lines.pop('', None)
                line = line_bridge.join(lines)
            else:
                line = lexer.get_line(first_tok)

        # Work out the highlight line:
        for t in between:
            token_length = len(t.value)
            kind = t.kind
            match kind:
                case kind if kind in (*T_Ignore, TokKind.NEWLINE, TokKind.EOF):
                    if t is between[-1]:
                        token_length = 0
                case TokKind.STRING:
                    # match_repr contains the quotation marks:
                    token_length = len(t.match_repr)
                case TokKind.UNKNOWN:
                    token_length = 1
                case _:
                    pass

            highlight += '^' * token_length

        # Determine how far along the first token is in the line:
        line_start = len(line) - len(line.lstrip())
        if between[-1].kind is TokKind.NEWLINE:
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
    Exception raised during string parsing operations.
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
    :type at: tuple[:class:`CharNo`, :class:`Location`]

    :param value: The literal text making up the token
    :type value: :class:`str`

    :param kind: The token's distict kind
    :type kind: :class:`TokKind`

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
        kind: TokKind,
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

    def __lt__(self, other) -> bool:
        pass
    def __le__(self, other) -> bool:
        pass
    def __gt__(self, other) -> bool:
        pass
    def __ge__(self, other) -> bool:
        pass

    def __class_getitem__(cls, item: TokKind) -> str:
        return cls
        item_name = (
            item.__class__.__name__
            if not hasattr(item, '__name__')
            else item.__name__
        )
        return f"{cls.__name__}[{item_name}]"

    def error_leader(self, with_col: bool = False) -> str:
        '''
        Return the beginning of an error message that features the
        filename, line number and possibly current column number.

        :param with_col: Also print the current column number,
            defaults to ``False``
        :type with_col: :class:`bool`
        '''
        file = f"File {self.file}, " if self.file is not None else ''
        column = ', column ' + str(self.colno) if with_col else ''
        msg = f"{file}Line {self.lineno}{column}: "
        return msg

    @property
    def match_repr(self) -> str | None:
        '''
        Return the token's value in quotes, if parsed as a string,
        else ``None``
        '''
        if not len(self.matchgroups) > 1:
            return None
        quote = self.matchgroups[0]
        val = self.value
        return f"{quote}{val}{quote}"

    @property
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
    SPEECH_CHARS = tuple('"\'`') + ('"""', "'''", "```")
    MULTILINE_QUOTES = SPEECH_CHARS[-3:]

    _rules = (
        # Data types

        ## Multiline strings
        (re.compile(
            r'^(?P<speech_char>["\'`]{3})'
            r'(?P<text>((?!(?P=speech_char)).|' + NEWLINE + r')*)'
            r'(?P=speech_char)'
        , re.MULTILINE), TokKind.STRING),

        ## Single-line strings
        (re.compile(
            r'^(?P<speech_char>["\'`])'
            r'(?P<text>((?!(?P=speech_char)).)*)'
            r'(?P=speech_char)'
        ), TokKind.STRING),

        ## Numbers
        (re.compile(
            r'^\d?(\d*|[\d_]*\d)'
            r'\.'
            # Match trailing '_' and catch resulting Py ValueError later:
            r'\d[\d_]*'
        ), TokKind.FLOAT),
            # Again here:
        (re.compile(r'^\d[\d_]*'), TokKind.INTEGER),

        ## Booleans
        (re.compile(r'^(true|yes)', re.IGNORECASE), TokKind.TRUE),
        (re.compile(r'^(false|no)', re.IGNORECASE), TokKind.FALSE),

        # Ignore
        ## Comments
        (re.compile(r'^\#.*(?=\n*)'), TokKind.COMMENT),
        ## Matching Newline helps find empty assignments:
        (re.compile(r'^' + NEWLINE + r'+'), TokKind.NEWLINE),
        ## All other whitespace
        (re.compile(r'^[^' + NEWLINE + r'\S]+'), TokKind.SPACE),

        # Operators
        (re.compile(r'^='), TokKind.EQUALS),
        (re.compile(r'^\.'), TokKind.ATTRIBUTE),

        # Syntax
        (re.compile(r'^\,'), TokKind.COMMA),
        (re.compile(r'^\('), TokKind.L_PAREN),
        (re.compile(r'^\)'), TokKind.R_PAREN),
        (re.compile(r'^\['), TokKind.L_BRACKET),
        (re.compile(r'^\]'), TokKind.R_BRACKET),
        (re.compile(r'^\{'), TokKind.L_CURLY_BRACE),
        (re.compile(r'^\}'), TokKind.R_CURLY_BRACE),

        # Symbols
        *((r'^' + kw, TokKind.KEYWORD) for kw in KEYWORDS),
        (re.compile(r'^[^' + NOT_IN_IDS + r']+'), TokKind.IDENTIFIER),
    )

    def __init__(
        self,
        string: str = None,
        file: str = None
    ) -> None: 
        self._string = string
        self._lines = self._string.split('\n')
        self._tokens = []
        self._string_stack = None
        self._token_stack = []

        self._cursor = 0  # 0-indexed
        self._lineno = 1  # 1-indexed
        self._colno = 1  # 1-indexed
        self.eof = TokKind.EOF
        self._file = file

    @property
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

    @property
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
        self._tokens = []
        return self

    def in_range(self, first: Token, last: Token) -> tuple[Token]:
        '''
        Gather tokens from between a range spanned by two tokens.

        :param first: The first token to gather
        :type first: :class:`Token`

        :param last: The last token to gather
        :type last: :class:`Token`
        '''
        tokens = self._tokens
        if last not in tokens and last.kind not in T_Invisible:
            tokens.append(last)
        start = first.cursor
        end = last.cursor
        between = tuple(t for t in tokens if start <= t.cursor <= end)
        return between

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
            return tuple(self._tokens[-back:])
        tokens = self._tokens
        idx = tuple(tok.cursor for tok in tokens).index(since.cursor)
        ret = tuple(tokens[idx - back : idx + 1])
        return ret

    def error_leader(self, with_col: bool = False) -> str:
        '''
        Return the beginning of an error message that features the
        filename, line number and possibly current column number.

        :param with_col: Also print the current column number,
            defaults to ``False``
        :type with_col: :class:`bool`
        '''
        # file = self._file if self._file is not None else ''
        column = ', column ' + str(self._colno) if with_col else ''
        # msg = f"File {file!r}, line {self._lineno}{column}: "
        msg = f"Line {self._lineno}{column}: "
        return msg

    def get_token(self) -> Token:
        '''
        Return the next token in the lexing stream.

        :raises: :exc:`TokenError` upon an unexpected token
        '''
        try:
            return next(self._get_token())
        except StopIteration as e:
            return e.value

    def _get_token(self) -> Token:
        '''
        A generator.
        Return the next token in the lexing stream.

        :raises: :exc:`StopIteration` to give the current token
        :raises: :exc:`TokenError` upon an unexpected token
        '''
        # The stack will have contents after string concatenation.
        if self._token_stack:
            tok = self._token_stack.pop()
            self._tokens.append(tok)
            return tok

        # Everything after and including the cursor position:
        s = self._string[self._cursor:]

        # Match against each rule:
        for test, kind in self._rules:
            m = re.match(test, s)

            if m is None:
                # This rule was not matched; try the next one.
                continue

            tok = Token(
                value=m.group(),
                at=(self._cursor, (self._lineno, self._colno)),
                kind=kind,
                matchgroups=m.groups(),
                lexer=self,
                file=self._file
            )

            if kind in T_Ignore:
                l = len(tok.value)
                self._cursor += l
                self._colno += l
                return (yield from self._get_token())

            # Update location:
            self._cursor += len(tok.value)
            self._colno += len(tok.value)
            if kind is TokKind.NEWLINE:
                self._lineno += len(tok.value)
                self._colno = 1

            if kind is TokKind.STRING:
                # Add to the string stack for error handling:
                self._string_debug = tok
                # Process strings by removing quotes:
                speech_char = tok.matchgroups[0]
                if speech_char in self.MULTILINE_QUOTES:
                    lines = tok.value.count(NEWLINE)
                    self._lineno += lines
                value = tok.value.strip(speech_char)
                if '\\' in value:
                    value = unescape_backslash(value)
                tok.value = value

                # Concatenate neighboring strings:
                if self.STRING_CONCAT:
                    while True:
                        maybe_str = (yield from self._get_token())
                        if maybe_str.kind in T_Ignore:
                            continue
                        break

                    if maybe_str.kind is TokKind.STRING:
                        # Concatenate.
                        tok.value += maybe_str.value
                        self._tokens.append(tok)
                        return tok

                    else:
                        # Handle the next token separately.
                        self._token_stack.append(maybe_str)

                # self._tokens.append(self._string_debug)
                self._string_debug = None

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
                kind=TokKind.UNKNOWN,
                matchgroups=None,
                lexer=self,
                file=self._file
            )
            bad_toks = (bad_token,)

            if bad_value.startswith(self.SPEECH_CHARS):
                if self._string_debug is None:
                    # Broken single speech characters
                    msg = f"Unmatched quote:"
                    raise TokenError.hl_error(bad_toks, msg)

                # Broken triple speech characters
                msg = f"Unmatched multiline quote:"
                self._tokens.append(self._string_debug)

                # Two quotes of each kind:
                pairs = (s*2 for s in self.SPEECH_CHARS[:3])
                maybe_first_two = self._string_debug
                if maybe_first_two.match_repr in pairs:
                    # The current token is part of a multiline speech
                    # char begun by the previous token.
                    bad_toks = (maybe_first_two, bad_token)

            else:
                too_long = 8
                if len(bad_value) > too_long:
                    msg = f"Unexpected token:"
                else:
                    msg = f"Unexpected token: {bad_value!r}"

            self._tokens.append(bad_token)

            # Do some stack trace magic:
            try:
                raise TokenError.hl_error(bad_toks, msg)
            except TokenError as e:
                # Avoid long stack traces with many frames:
                import __main__ as possibly_repl
                if not hasattr(possibly_repl, '__file__'):
                    # User is in a REPL! Don't yeet them.
                    raise
                else:
                    # OK, yeet:
                    import traceback
                    traceback.print_exc(limit=1)
                    raise StackTraceSilencer(1)  # Sorry...


class RecursiveDescentParser:
    '''
    Parse strings, converting them to abstract syntax trees.
    Use generators and stacks to overcome Python's recursion limit.

    :param lexer: The lexer used for feeding tokens
    :type lexer: :class:`Lexer`

    :param string: A string to parse if `lexer` is not given
    :type string: :class:`FileContents`
    '''
    def __init__(
        self,
        lexer: Lexer = None,
        string: FileContents = None
    ) -> None:
        if lexer is None:
            if string is None:
                msg = (
                    "A `string` argument is required when `lexer` is"
                    " not given."
                )
                raise ValueError(msg)
            lexer = Lexer(string)
        self._lexer = lexer
        self._string = string
        self._tokens = self._lexer.lex()
        self._cursor = 0
        self._expr_stack = []

    End = TokKind.EOF

    @property
    def lookahead(self) -> Token:
        '''
        Return the current token.
        :rtype: :class:`Token`
        '''
        return self._tokens[self._cursor]

    @property
    def tokens(self) -> list[Token]:
        '''
        Return the list of tokens generated by the lexer.
        :rtype: list[:class:`Token`]
        '''
        return self._tokens

    def _token_dict(self) -> dict[CharNo, Token]:
        '''
        Return a dict mapping lexer cursor value to tokens.
        :rtype: dict[:class:`CharNo`, :class:`Token`]
        '''
        return {t.cursor: t for t in self.tokens()}

    def at_eof(self) -> bool:
        '''
        Return whether the parser has reached the end of the token stream.
        :rtype: :class:`bool`
        '''
        return self._cursor == len(self._tokens) - 1

    def _advance(self) -> Token:
        '''
        Move to the next token. Return that token.
        :rtype: :class:`Token`
        '''
        if self._cursor < len(self._tokens) - 1:
            self._cursor += 1
        return self._tokens[self._cursor]

    def _skip_all_whitespace(self) -> Token:
        '''
        Return the current or next non-whitespace token.
        This also skips newlines.
        :rtype: :class:`Token`
        '''
        tok = self._tokens[self._cursor]
        while True:
            if tok.kind not in (*T_Ignore, TokKind.NEWLINE):
                return tok
            tok = self._advance()

    def _next(self) -> Token:
        '''
        Advance to the next non-whitespace token. Return that token.
        :rtype: :class:`Token`
        '''
        while True:
            if not (tok := self._advance()).kind in T_Ignore:
                return tok

    def _match(
        self,
        kind: TokKind | Iterable[TokKind],
        errmsg: str = None,
        greedy: bool = False
    ) -> Token | NoReturn | bool:
        '''
        Test if the current token matches a certain kind given by `kind`.
        If the test fails, raise :exc:`ParseError` using
        `errmsg` or return ``False`` if `errmsg` is ``None``.
        If the test passes, return the current token.

        :param kind: The kind(s) to expect from the current token
        :type kind: :class:`TokKind`

        :param errmsg: The error message to display, optional
        :type errmsg: :class:`str`

        :param greedy: Consume all matching tokens and return the first
            token that does not match.
        :type greedy: :class:`bool`

        :raises: :exc:`ParseError` When the test fails, `errmsg` is
            given and `greedy` is ``False`` 
        '''
        if not isinstance(kind, Iterable):
            kind = (kind,)
        tok = self.lookahead
        if greedy:
            while tok.kind in kind:
                tok = self._advance()
            return tok
        if tok.kind not in kind:
            if errmsg is None:
                return False
            raise ParseError.hl_error(tok, errmsg, self)
        self._advance()
        return tok

    @property
    def prev_tok(self) -> Token:
        '''
        Return the last parsed token.
        '''
        return self._tokens[self._cursor - 1]

    def _reset(self) -> None:
        '''
        Return the lexer to the first token of the token stream.
        '''
        self._cursor = 0
        self._lexer.reset()

    def _parse(self, callback: Callable) -> AST:
        '''
        Use generators and a stack to bypass the Python recursion limit.
        Credit to Dave Beazley (2014).
        Run `callback` outside the Python call stack.

        :param callback: The generator to run
        :type callback: :class:`Callable`
        '''
        stack = [self._gen_parse(callback)]
        result = None
        while stack:
            try:
                callback = stack[-1].send(result)
                stack.append(self._gen_parse(callback))
                result = None
            except StopIteration as e:
                stack.pop()
                result = e.value
        return result

    def _gen_parse(self, callback: Callable) -> AST:
        '''
        Use generators and a stack to bypass the Python recursion limit.
        Credit to Dave Beazley (2014).
        Run `callback` outside the Python call stack by sending it to
        `_parse`.

        :param callback: The generator to send
        :type callback: :class:`Callable`
        '''
        if not callable(callback):
            return callback
        result = callback()
        if isinstance(result, GeneratorType):
            result = (yield from result)
        return result

    def _parse_Assignment(self) -> Assign | None:
        self._skip_all_whitespace()
        if self.at_eof():
            return TokKind.EOF
        target = (yield self._parse_Identifier)
        self._parse_MaybeEQ()
        value = (yield self._parse_Expr)
        if value is TokKind.NEWLINE:
            value = Constant(None)
        self._parse_Delimiter()
        node = Assign([target], value)
        node._token = target._token
        return node

    def _parse_Identifier(self) -> Name | Attribute:
        base = self.lookahead
        kind = base.kind
        node = Name(base.value)
        node._token = base

        note = ""
        if kind is not TokKind.IDENTIFIER:
            if kind in T_Syntax:
                if self._expr_stack:
                    curr_expr = self._expr_stack[-1]
                    suggestion = {
                        Dict: '}',
                        List: ']',
                    }[curr_expr]
                    note = f"Did you mean to use {suggestion!r}? "
                else:
                    note = f"Unmatched {kind.value!r} "

        msg = f"Invalid syntax: {note}(Expected an identifier):"
        self._match(TokKind.IDENTIFIER, msg)

        # This may be the base of another attr, or it may be the
        # terminal attr:
        while self._match(TokKind.ATTRIBUTE):
            maybe_base_tok = self._match(TokKind.IDENTIFIER, msg)
            attr = maybe_base_tok.value
            node = Attribute(node, attr)
            node._token = maybe_base_tok
        return node

    def _parse_MaybeEQ(self) -> MaybeEQ:
        self._match(MatchesMaybeEQ)

    def _parse_Delimiter(self) -> Delimiter:
        self._match(MatchesDelimiter, greedy=True)

    def _parse_Expr(self) -> Expr:
        if self._match(T_AssignEvalNone):
            return Constant(None)

        tok = self._match(T_Literal)
        if tok:
            try:
                match tok.kind:
                    case TokKind.STRING:
                        value = tok.value
                    case TokKind.INTEGER:
                        value = int(tok.value)
                    case TokKind.FLOAT:
                        value = float(tok.value)
                    case TokKind.TRUE:
                        value = True
                    case TokKind.FALSE:
                        value = False
                    case TokKind.NONE:
                        value = None
                    case _:
                        msg = f"Expected a literal, but got {tok.value!r}"
                        raise ParseError.hl_error(tok, msg, self)
                node = Constant(value)
            except ValueError:
                # Floats with underscores sometimes fail.
                msg = f"Invalid {kind.value.lower()} value: {tok.value!r}"
                raise ParseError.hl_error(tok, msg, self)

        else:
            # This is a complex expression.
            tok = self.lookahead
            kind = tok.kind
            if kind is TokKind.IDENTIFIER:
                node = (yield self._parse_Identifier)
            elif kind is TokKind.L_BRACKET:
                node = (yield self._parse_List)
            elif kind is TokKind.L_CURLY_BRACE:
                node = (yield self._parse_Dict)
            else:
                msg = (f"Expected an expression,"
                       f" but got {tok.value!r} instead.")
                raise ParseError.hl_error(tok, msg, self)

        node._token = tok
        self._skip_all_whitespace()
        return node

    def _parse_List(self) -> List:
        self._expr_stack.append(List)
        msg = "_parse_List() called at the wrong time"
        start = self._match(TokKind.L_BRACKET, msg)
        elems = (yield self._parse_RepeatedExpr)
        self._match(TokKind.R_BRACKET)
        node = List(elems)
        node._token = start
        self._expr_stack.pop()
        return node

    def _parse_RepeatedExpr(self) -> RepeatedExpr:
        start = self.prev_tok
        exprs = []
        while True:
            self._skip_all_whitespace()
            if self.lookahead.kind is TokKind.EOF:
                msg = "Invalid syntax: Unmatched '[':"
                raise ParseError.hl_error(start, msg, self)
            if self.lookahead.kind is TokKind.R_BRACKET:
                break
            expr = (yield self._parse_Expr)
            # Match commas and newlines:
            self._parse_Delimiter()
            exprs.append(expr)
        return exprs

    def _parse_Dict(self) -> Dict:
        self._expr_stack.append(Dict)
        msg = "_parse_Dict() called at the wrong time",
        start = self._match(TokKind.L_CURLY_BRACE, msg)
        keys, vals = (yield self._parse_RepeatedKVP)
        self._match(TokKind.R_CURLY_BRACE)
        node = Dict(keys, vals)
        node._token = start
        self._expr_stack.pop()
        return node

    def _parse_RepeatedKVP(self) -> RepeatedKVP:
        start = self.prev_tok
        keys = []
        vals = []
        while True:
            self._skip_all_whitespace()
            if self.lookahead.kind is TokKind.EOF:
                msg = "Invalid syntax: Unmatched '{':"
                raise ParseError.hl_error(start, msg, self)
            if self.lookahead.kind is TokKind.R_CURLY_BRACE:
                break
            pair = (yield self._parse_KVPair)
            self._parse_Delimiter()
            key, val = pair

            if key not in keys:
                keys.append(key)
                vals.append(val)
            else:
                # Make nested dicts.
                idx = keys.index(key)
                # Equivalent to dict.update():
                vals[idx].keys = key
                vals[idx].values = val
        return keys, vals

    def _parse_KVPair(self) -> KVPair:
        target = (yield self._parse_Identifier)
        self._parse_MaybeEQ()
        value = (yield self._parse_Expr)
        # Disallow assigning identifiers:
        if isinstance(value, (Name, Attribute)):
            typ = value._token.kind.value.lower()
            val = repr(value._token.value)
            note = f"expected expression, got {typ} {val}"
            msg = f"Invalid assignment: {note}:"
            raise ParseError.hl_error(value._token, msg, self)
        return target, value

    def parse(self, string: FileContents = None) -> Module:
        '''
        Parse the lexer stream and return it as a :class:`Module`.

        :param string: A new string to use for parsing, optional
        :type string: :class:`FileContents`
        '''
        if string is not None:
            self._string = string
            self._lexer = Lexer(string)
            self._tokens = self._lexer.lex()
            self._cursor = 0

        self._reset()
        assignments = []
        while True:
            assign = self._parse(self._parse_Assignment)
            if assign is TokKind.EOF:
                break
            if assign is None:
                # This happens when there are no assignments at all.
                break
            assignments.append(assign)
        self._reset()
        return Module(assignments)

    def to_dict(self) -> dict[str]:
        '''
        Parse a config file and return its data as a :class:`dict`.
        '''
        tree = self.parse()
        unparsed = Unparser().unparse(tree)
        # print(unparsed)
        return unparsed


class FileParser(RecursiveDescentParser):
    '''
    A class for parsing text files.

    :param file: The filename used to source an input string
    :type file: :class:`PathLike`

    :param lexer: The lexer to use for parsing.
    :type lexer: :class:`Lexer`
    '''
    def __init__(
        self,
        file: PathLike = None,
        *,
        lexer: Lexer = None,
    ) -> None:
        if lexer is None:
            if file is None:
                msg = "Either a `file` or a `lexer` argument is required"
                raise ValueError(msg)
            with open(file, 'r') as f:
                string = f.read()
            lexer = Lexer(string, file)

        self._file = file
        self._string = string
        self._lexer = lexer
        self._tokens = self._lexer.lex()
        self._cursor = 0
        self._expr_stack = []


class PyParser:
    '''
    Convert Python config data to ASTs.
    '''

    ELEMS_ARE_FIELD_NAMES = ('field_order',)

    @classmethod
    def parse(cls, data: Mapping | list[dict[str]]) -> Module:
        '''
        Convert a root-level config dict to a Module AST.

        :param data: The dict to convert
        :type data: :class:`Mapping`
        '''
        assignments = []
        for key, val in data.items():

            # Treat to field names:
            if key in cls.ELEMS_ARE_FIELD_NAMES:
                if isinstance(val, Sequence):
                    val = List([Name(field) for field in val])
                    a = Assign([Name(key)], val)

            else:
                a = Assign([Name(key)], cls._parse_node(val))
            assignments.append(a)
        return Module(assignments)

    @classmethod
    def make_file(cls, data: Mapping) -> FileContents:
        '''
        Convert a Python dictionary to an AST and return the contents of
        the config file that could be parsed back into that AST.

        :param data: The dict to convert
        :type data: :class:`Mapping`
        '''
        tree = cls.parse(data)
        text = ConfigFileMaker().stringify(tree)
        return text

    @classmethod
    def _run_process_nested_dict(
        cls,
        dct: dict | Any,
        roots: Sequence[str] = [],
        descended: int = 0
    ) -> tuple[list[list[str]], list[PythonData]]:
        '''
        Unravel nested dicts into keys and assignment values.
        This method runs :meth:`_process_nested_dict()`
        For use with :meth:`_process_nested_attrs()`
        Use generators and a stack to bypass the Python recursion limit.

        :param dct: The nested dict to process, or an inner value of it
        :type dct: :class:`dict` | :class:`Any`

        :param roots: The keys which surround the current `dct`
        :type roots: :class:`Sequence`[:class:`str`], defaults to ``[]``

        :param descended: How far `dct` is from the outermost key
        :type descended: :class:`int`, defaults to ``0``
        '''
        stack = [cls._process_nested_dict(dct, roots, descended)]
        result = None
        while stack:
            try:
                args = stack[-1].send(result)
                stack.append(cls._process_nested_dict(*args))
                result = None
            except StopIteration as e:
                stack.pop()
                result = e.value
        return result

    @staticmethod
    def _process_nested_dict(
        dct: Mapping | Any,
        roots: Sequence[str] = [],
        descended: int = 0
    ) -> tuple[list[list[str]], list[PythonData]]:
        '''
        Unravel nested dicts into keys and assignment values.
        For use with :meth:`_process_nested_attrs()`

        :param dct: The nested dict to process, or an inner value of it
        :type dct: :class:`dict` | :class:`Any`

        :param roots: The keys which surround the current `dct`
        :type roots: :class:`Sequence`[:class:`str`], defaults to ``[]``

        :param descended: How far `dct` is from the outermost key
        :type descended: :class:`int`, defaults to ``0``

        :returns: Attribute names and assignment values
        '''
        nodes = []
        vals = []
        if not isinstance(dct, dict):
            # An assignment.
            return ([roots], [dct])

        if len(dct) == 1:
            # An attribute.
            for a, v in dct.items():
                roots.append(a)
                result = (yield (v, roots, descended))
                return result

        descended = 0  # Start of a tree.
        for attr, v in dct.items():
            roots.append(attr)
            if isinstance(v, dict):
                if descended < len(roots):
                    descended = -len(roots)
                # Descend into lower tree.
                inner = (yield (v, roots, descended))
                if isinstance(inner, GeneratorType):
                    inner = (yield from inner)
                inner_nodes, inner_vals = inner
                nodes.extend(inner_nodes)
                vals.extend(inner_vals)
            else:
                nodes.append(roots)
                vals.append(v)
            roots = roots[:-descended - 1]  # Reached end of tree.
        return nodes, vals

    @staticmethod
    def _process_nested_attrs(
        attrs: Sequence[Sequence[str]]
    ) -> list[Attribute]:
        '''
        Convert a list of attribute names to an :class:`Attribute` node.

        :param attrs: The list of attribute keynames to convert
        :type attrs: :class:`Sequence`[:class:`Sequence`[:class:`str`]]
        '''
        nodes = []
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
    def _parse_node(cls, node: Any) -> AST:
        '''
        Parse a Python object and return its corresponding AST node.

        :param node: The node to parse
        :type node: :class:`Any`
        '''
        if isinstance(node, Mapping):
            keys = []
            vals = []
            for key, val in node.items():
                if isinstance(val, Mapping):
                    if len(node) == 1:
                        attribute = True
                    else:
                        attribute = False

                    # An attribute, possibly nested, or an assignment.
                    attrs, assigns = cls._run_process_nested_dict(val, [key])
                    nodes = cls._process_nested_attrs(attrs)
                    keys.extend(nodes)
                    vals.extend(assigns)

                else:
                    # An explicit assignment.
                    attribute = False
                    keys.append(Name(key))
                    v = cls._parse_node(val)
                    vals.append(v)

            if attribute:
                if len(vals) == 1:
                    return Assign(keys, cls._parse_node(*vals))
            return Dict(keys, [cls._parse_node(v) for v in vals])

        elif isinstance(node, list):
            values = [cls._parse_node(v) for v in node]
            return List(values)

        elif isinstance(node, (int, float, str)):
            return Constant(node)

        elif node in (None, True, False):
            return Constant(node)

        else:
            return node


class Unparser(NodeVisitor):
    '''
    Convert ASTs to Python data structures.
    '''
    _EXPR_PLACEHOLDER = '_EXPR_PLACEHOLDER'

    def unparse(self, node: AST) -> PythonData:
        '''
        Unparse an AST, converting it to an equivalent Python data
        structure.
        '''
        return self.visit(node)

    def visit(self, node: AST):
        '''
        Use a stack to overcome the Python recursion limit.
        Credit to Dave Beazley (2014).
        '''
        stack = [self.genvisit(node)]
        result = None
        while stack:
            try:
                node = stack[-1].send(result)
                stack.append(self.genvisit(node))
                result = None
            except StopIteration as e:
                stack.pop()
                result = e.value
        return result

    def genvisit(self, node: AST):
        '''
        Use generators to overcome the Python recursion limit.
        Credit to Dave Beazley (2014).
        '''
        name = 'visit_' + type(node).__name__
        result = getattr(self, name)(node)
        if isinstance(result, GeneratorType):
            result = (yield from result)
        return result

    def visit_Assign(self, node: Assign) -> dict:
        target = (yield (node.targets[-1]))
        value = (yield (node.value))
        if not isinstance(target, str):
            # `target` is an attribute turned into a nested dict
            return self._run_nested_update({}, target, value)
        return {target: value}

    def visit_Attribute(self, node: Attribute) -> dict:
        attrs = self._nested_attr_to_dict(node)
        new_d = self._EXPR_PLACEHOLDER
        for attr in attrs:
            new_d = {attr: new_d}
        return new_d

    @staticmethod
    def _nested_attr_to_dict(
        node: Attribute | Name,
        attrs: Sequence[str] = []
    ) -> list[str]:
        '''
        Convert nested :class:`Attribute`s to a list of attr names.

        :param node: The Attribute to convert
        :type node: :class:`Attribute` | :class:`Name`

        :param attrs: A list of preexisting attribute names
        :type attrs: :class:`Sequence`[:class:`str`]
        '''
        while not isinstance(node, Name):
            if not attrs:
                # A new nested Attribute.
                attrs = [node.attr]
            elif isinstance(node, Attribute):
                attrs.append(node.attr)
            node = node.value

        attrs.append(node.id)
        return attrs

    def visit_Constant(self, node: Constant) -> ConstantValue:
        return node.value

    def _run_nested_update(
        self,
        orig: Mapping | Any,
        upd: Mapping,
        assign: Any = {}
    ) -> dict:
        '''
        Merge two dicts and properly give them their innermost values.
        Use generators and a stack to bypass the Python recursion limit.
        Credit to Dave Beazley (2014).

        :param orig: The original dict
        :type orig: :class:`dict` | :class:`Any`

        :param upd: A dict that updates `orig`
        :type upd: :class:`dict`

        :param assign: A value that will be stored in the innermost dict,
            defaults to ``{}`` for continued attribute parsing.
        :type assign: :class:`Any`
        '''
        stack = [self._nested_update(orig, upd, assign)]
        result = None
        while stack:
            try:
                args = stack[-1].send(result)
                stack.append(self._nested_update(*args))
                result = None
            except StopIteration as e:
                stack.pop()
                result = e.value
        return result

    def _nested_update(
        self,
        orig: dict | Any,
        upd: dict,
        assign: Any = {}
    ) -> dict:
        '''
        Merge two dicts and properly give them their innermost values.
        Use generators and a stack to bypass the Python recursion limit.
        Credit to Dave Beazley (2014).

        :param orig: The original dict
        :type orig: :class:`dict` | :class:`Any`

        :param upd: A dict that updates `orig`
        :type upd: :class:`dict`

        :param assign: A value that will be stored in the innermost dict,
            defaults to ``{}`` for continued attribute parsing.
        :type assign: :class:`Any`
        '''
        for k, v in upd.items():
            if v is self._EXPR_PLACEHOLDER:
                v = assign
            # We cannot use Mapping because its instance check is recursive.
            if not isinstance(orig, dict):
                # Replace the old value with the new:
                return upd

            if isinstance(v, dict):
                # Give parameters to the generator runner:
                updated = (yield (orig.get(k, {}), v, assign))
                if isinstance(updated, GeneratorType):
                    updated = (yield from updated)
                orig[k] = updated
            else:
                orig[k] = v
        return orig

    def visit_Dict(self, node: Dict) -> dict[str]:
        new_d = {}
        for key, val in zip(node.keys, node.values):
            target = (yield key)
            value = (yield val)
            if isinstance(target, dict):
                # An Attribute
                new_d = self._run_nested_update(new_d, target, value)
            else:
                # An assignment
                new_d.update({target: value})
        return new_d

    def visit_List(self, node: List) -> list:
        l = []
        for e in node.elts:
            l.append((yield e))
        return l

    def visit_Module(self, node: Module) -> dict[str]:
        config = {}
        for n in node.body:
            assignment = (yield n)
            if isinstance(n, Assign):
                target_node = n.targets[0]
                # Handle dict updates made using attributes:
                if isinstance(target_node, Attribute):
                    (target_name, new) ,= assignment.items()
                    if target_name in config and isinstance(new, dict):
                        old = config[target_name]
                        updated = self._run_nested_update(old, new)
                        assignment = {target_name: updated}
            config.update(assignment)
        return config

    def visit_Name(self, node: Name) -> str:
        return node.id


class ConfigFileMaker(NodeVisitor):
    '''
    Convert ASTs to string representations with config file syntax.
    '''
    def __init__(self) -> None:
        self._indent = 4 * ' '
        self._indent_level = 0

    def indent(self) -> str:
        return self._indent_level * self._indent

    def stringify(
        self,
        tree: Iterable[AST] | Module,
        sep: str = '\n\n'
    ) -> FileContents:
        '''
        Convert an AST to the contents of a config file.

        :param tree: The ASTs to be converted to strings
        :type tree: :class:`Iterable`[:class:`AST`] | :class:`Module`

        :param sep: The separator to use between tree nodes,
            defaults to ``'\n\n'``
        :type sep: :class:`str`
        '''
        if isinstance(tree, Module):
            tree = tree.body
        strings = (self.visit(node) for node in tree)
        return sep.join(strings)

    def visit(self, node: AST):
        '''
        Use a stack to overcome the Python recursion limit.
        Credit to Dave Beazley (2014).
        '''
        stack = [self.genvisit(node)]
        result = None
        while stack:
            try:
                node = stack[-1].send(result)
                stack.append(self.genvisit(node))
                result = None
            except StopIteration as e:
                stack.pop()
                result = e.value
        return result

    def genvisit(self, node: AST):
        '''
        Use generators to overcome the Python recursion limit.
        Credit to Dave Beazley (2014).
        '''
        name = 'visit_' + type(node).__name__
        result = getattr(self, name)(node)
        if isinstance(result, GeneratorType):
            result = (yield from result)
        return result

    def visit_Attribute(self, node: Attribute) -> str:
        base = (yield node.value)
        attr = node.attr
        return f"{base}.{attr}"

    def visit_Assign(self, node: Assign) -> str:
        target = (yield node.targets[-1])
        value = (yield node.value)
        return f"{target} {value}"

    def visit_Constant(self, node: Constant) -> str:
        val = node.value
        if isinstance(val, str):
            return repr(val)
        if val is None:
            return ""
        return str(val)

    def visit_Dict(self, node: Dict) -> str:
        keys = []
        values = []
        for k in node.keys:
            keys.append((yield k))
        for v in node.values:
            self._indent_level += 1
            values.append((yield v))
            self._indent_level -= 1
        assignments = tuple(' '.join(pair) for pair in zip(keys, values))
        self._indent_level += 1
        if len(assignments) > 1:
            opening = f"{{\n{self.indent()}"
            joiner = f"\n{self.indent()}" 
            self._indent_level -= 1
            closing = f"\n{self.indent()}}}"
        else:
            opening = "{"
            joiner = ""
            closing = "}"
        joined = joiner.join(assignments)
        string = f"{opening}{joined}{closing}"
        self._indent_level -= 1
        return string

    def visit_List(self, node: List) -> str:
        one_line_limit = 3
        elems = []
        for e in node.elts:
            elems.append((yield e))
        if len(elems) > one_line_limit:
            self._indent_level += 1
            joined = ('\n' + self.indent()).join(elems)
            string = f"[\n{self.indent()}{joined}\n]"
            self._indent_level -= 1
        else:
            joined = ', '.join(str(e) for e in elems)
            string = f"[{joined}]"
        return string

    def visit_Name(self, node: Name) -> str:
        string = node.id
        return string


def parse(file: PathLike = None, string: FileContents = None) -> Module:
    '''
    Parse a config file and return its AST.

    :param file: The file to parse
    :type file: :class:`PathLike`
    '''
    if string is None:
        return FileParser(file).parse()
    if file is None:
        return RecursiveDescentParser(Lexer(string)).parse()
    raise ValueError(
        "A `file` argument is required when `string` is not given or None."
    )


def ast_to_data(node: AST) -> PythonData:
    '''
    Convert an AST to Python data.

    :param node: The AST to convert
    :type node: :class:`AST`
    '''
    return Unparser().unparse(node)


def data_to_text(data: PythonData) -> FileContents:
    '''
    Convert Python data to config file text.

    :param data: The data to convert
    :type data: :class:`PythonData`
    '''
    return PyParser.make_file(data)


def text_to_data(contents: FileContents) -> PythonData:
    '''
    Convert config file text to Python data.

    :param contents: The text to parse
    :type contents: :class:`FileContents`
    '''
    module = RecursiveDescentParser(string=contents).parse()
    return Unparser().unparse(module)


def convert_file(file: PathLike = None) -> PythonData:
    '''
    Read a config file, parse its contents and return the encoded data.

    :param file: The file to parse,
        defaults to :obj:`mybar.constants.CONFIG_FILE`
    '''
    if file is None:
        file = CONFIG_FILE
    absolute = os.path.abspath(os.path.expanduser(file))
    module = FileParser(absolute).parse()
    return Unparser().unparse(module)

