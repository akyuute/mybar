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
from typing import Any, Literal, NamedTuple, NoReturn, Self, TypeAlias, TypeVar

# from ._types import FileContents, PythonData
FileContents = 'FileContents'
PythonData = 'PythonData'

Token = TypeVar('Token')
Lexer = TypeVar('Lexer')


CharNo: TypeAlias = int
LineNo: TypeAlias = int
ColNo: TypeAlias = int
TokenValue: TypeAlias = str
Location: TypeAlias = tuple[LineNo, ColNo]


EOF = ''
NEWLINE = '\n'
UNKNOWN = 'UNKNOWN'
NOT_IN_IDS = string.punctuation.replace('_', '\s')

class Grammar:
    '''
    Module : Assignment
           | Assignment Delimiter Module

    Assignment : IDENTIFIER MaybeEQ Expr

    Expr : LITERAL
         | List
         | Dict
         | Delimiter
         | None

    List : L_BRACKET RepeatedExpr R_BRACKET

    Dict : L_CURLY_BRACE RepeatedKVP R_CURLY_BRACE

    RepeatedExpr : Expr Delimiter RepeatedExpr
                 | None

    RepeatedKVP : KVPair Delimiter RepeatedKVP
                | None

    KVPair : IDENTIFIER MaybeEQ Expr

    MaybeEQ : EQUALS | None

    Delimiter : NEWLINE | COMMA | None
    '''


class DictImpliedAssign(Assign):
    pass

class Newline(Enum):
    NEWLINE = repr(NEWLINE)

class EndOfFile(Enum):
    EOF = repr(EOF)

class T_Literal(Enum):
    FLOAT = 'FLOAT'
    INTEGER = 'INTEGER'
    STRING = 'STRING'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    NONE = 'NONE'

class T_Ignore(Enum):
    COMMENT = 'COMMENT'
    SPACE = 'SPACE'

class T_Symbol(Enum):
    IDENTIFIER = 'IDENTIFIER'
    KEYWORD = 'KEYWORD'

class T_Keyword(Enum):
    pass
KEYWORDS = ()

class T_BinOp(Enum):
    ASSIGN = '='
    ATTRIBUTE = '.'
    ADD = '+'
    SUBTRACT = '-'

class T_Syntax(Enum):
    COMMA = ','
    L_PAREN = '('
    R_PAREN = ')'
    L_BRACKET = '['
    R_BRACKET = ']'
    L_CURLY_BRACE = '{'
    R_CURLY_BRACE = '}'

class T_Unknown(Enum):
    UNKNOWN = 'UNKNOWN'

# T_AssignEvalNone = tuple((*Newline, *EndOfFile))
T_AssignEvalNone = (Newline.NEWLINE, EndOfFile.EOF)
'''These tokens eval to None after an equals sign.'''

TokenKind = Enum(
    'TokenKind',
    ((k.name, k.value) for k in (
        *T_Literal,
        *T_Symbol,
        *T_Syntax,
        *T_BinOp,
        *T_Ignore,
        *T_AssignEvalNone,
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
            if first_tok.kind is T_Literal.STRING:
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
            kind = t.kind
            match kind:

                case kind if kind in (*T_Ignore, EndOfFile.EOF):
                    if t is between[-1]:
                        # highlight += '^'
                        token_length = 1

                case T_Literal.STRING:
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
    :type at: tuple[:class:`CharNo`, :class:`Location`]

    :param value: The literal text making up the token
    :type value: :class:`str`

    :param kind: The token's distict kind
    :type kind: :class:`TokenKind`

    :param matchgroups: The value gotten by re.Match.groups() when
        making this token
    :type matchgroups: tuple[:class:`str`]
    '''

##    :param lexer: The lexer used to find this token, optional
##    :type lexer: :class:`Lexer`
##
##    :param file: The file from which this token came, optional
##    :type file: :class:`PathLike`
##    '''
    __slots__ = (
        'at',
        'value',
        'kind',
        'matchgroups',
        'cursor',
        'lineno',
        'colno',
        # 'lexer',
        # 'file',
    )

    def __init__(
        self,
        at: tuple[CharNo, Location],
        value: TokenValue,
        kind: TokenKind,
        matchgroups: tuple[str],
        # lexer: Lexer = None,
        # file: PathLike = None,
    ) -> None:
        self.at = at
        self.value = value
        self.kind = kind
        self.matchgroups = matchgroups
        self.cursor = at[0]
        self.lineno = at[1][0]
        self.colno = at[1][1]
        # self.lexer = lexer
        # self.file = file

    def __repr__(self):
        cls = type(self).__name__
        # ignore = ('matchgroups', 'cursor', 'lineno', 'colno', 'lexer', 'file')
        ignore = ('matchgroups', 'cursor', 'lineno', 'colno')
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

##    def error_leader(self, with_col: bool = False) -> str:
##        '''
##        Return the beginning of an error message that features the
##        filename, line number and possibly current column number.
##
##        :param with_col: Also print the token's column number,
##            defaults to ``False``
##        :type with_col: :class:`bool`
##        '''
##        file = f"File {self.file}, " if self.file is not None else ""
##        column = ', column ' + str(self.colno) if with_col else ""
##        msg = f"{file}line {self.lineno}{column}: "
##        return msg

    def coords(self) -> Location:
        '''
        Return the token's current coordinates as (line, column).
        '''
        return (self.lineno, self.colno)
    

class Lexer:
    '''
    The lexer splits text apart into individual tokens.

    :param string: If not using a file, use this string for lexing.
    :type string: :class:`str`
    '''

##    :param file: The file to use for lexing.
##        When unset or ``None``, use `string` by default.
##    :type file: :class: `PathLike`
##    '''
    STRING_CONCAT = True  # Concatenate neighboring strings
    SPEECH_CHARS = tuple('"\'`') + ('"""', "'''")

    _rules = (
        # Data types
        (re.compile(
            r'^(?P<speech_char>["\'`]{3}|["\'`])'
            r'(?P<text>(?!(?P=speech_char)).*?)*'
            r'(?P=speech_char)'
        ), T_Literal.STRING),
        (re.compile(r'^\d*\.\d[\d_]*'), T_Literal.FLOAT),
        (re.compile(r'^\d+'), T_Literal.INTEGER),

        # Ignore
        ## Skip comments
        (re.compile(r'^\#.*(?=\n*)'), T_Ignore.COMMENT),
        ## Finds empty assignments:
        (re.compile(r'^' + NEWLINE + r'+'), Newline.NEWLINE),
        ## Skip all other whitespace:
        (re.compile(r'^[^' + NEWLINE + r'\S]+'), T_Ignore.SPACE),

        # Operators
        (re.compile(r'^='), T_BinOp.ASSIGN),
        (re.compile(r'^\.'), T_BinOp.ATTRIBUTE),

        # Syntax
        (re.compile(r'^\,'), T_Syntax.COMMA),
        (re.compile(r'^\('), T_Syntax.L_PAREN),
        (re.compile(r'^\)'), T_Syntax.R_PAREN),
        (re.compile(r'^\['), T_Syntax.L_BRACKET),
        (re.compile(r'^\]'), T_Syntax.R_BRACKET),
        (re.compile(r'^\{'), T_Syntax.L_CURLY_BRACE),
        (re.compile(r'^\}'), T_Syntax.R_CURLY_BRACE),

        # Booleans
        (re.compile(r'^(true|yes)', re.IGNORECASE), T_Literal.TRUE),
        (re.compile(r'^(false|no)', re.IGNORECASE), T_Literal.FALSE),

        # Symbols
        *tuple(dict.fromkeys(KEYWORDS, T_Symbol.KEYWORD).items()),
        (re.compile(r'^[^' + NOT_IN_IDS + r']+'), T_Symbol.IDENTIFIER),
    )

    def __init__(
        self,
        string: str = None,
        # file: PathLike = None,
    ) -> None: 
##        if string is None:
##            if file is None:
##                msg = "`string` parameter is required when `file` is not given"
##                raise ValueError(msg)
##
##            with open(file, 'r') as f:
##                string = f.read()

        self._string = string
        self._lines = self._string.split('\n')
        # self._file = file
        self._tokens = []
        self._string_stack = LifoQueue()

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
        # file = self._file if self._file is not None else ''
        column = ', column ' + str(self._colno) if with_col else ''
        msg = f"File {file!r}, line {self._lineno}{column}: "
        return msg

    def get_token(self) -> Token:
        '''
        Return the next token in the lexing stream.

        :raises: :exc:`TokenError` upon an unexpected token
        '''
        # The stack will have contents after string concatenation.
        if not self._string_stack.empty():
            tok = self._string_stack.get()
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
                # lexer=self,
                # file=self._file
            )

            if kind in (T_Ignore.SPACE, T_Ignore.COMMENT):
                l = len(tok.value)
                self._cursor += l
                self._colno += l
                return tok

            # Update location:
            self._cursor += len(tok.value)
            self._colno += len(tok.value)
            if kind is Newline.NEWLINE:
                self._lineno += len(tok.value)
                self._colno = 1

            if kind is T_Literal.STRING:
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
                        if maybe_str.kind in (*T_Ignore, Newline.NEWLINE):
                            continue
                        break

                    if maybe_str.kind is T_Literal.STRING:
                        # Concatenate.
                        tok.value += maybe_str.value
                        self._tokens.append(tok)
                        return tok

                    else:
                        # Handle the next token separately.
                        self._string_stack.put(maybe_str)

            self._tokens.append(tok)
            return tok

        else:
            if s is EOF:
                tok = Token(
                    value=s,
                    at=(self._cursor, (self._lineno, self._colno)),
                    kind=self.eof,
                    matchgroups=None,
                    # lexer=self,
                    # file=self._file
                )

                self._tokens.append(tok)
                return tok

            # If a token is not returned, prepare an error message:
            bad_value = s.split(None, 1)[0]
            bad_token = Token(
                value=bad_value,
                at=(self._cursor, (self._lineno, self._colno)),
                kind=UNKNOWN,
                matchgroups=None,
                # lexer=self,
                # file=self._file
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


class LLParser:
    '''
    Parse config files, converting them to abstract syntax trees.

    :param lexer: The lexer used for feeding tokens
    :type lexer: :class:`Lexer`

##    :param file: If given, parse this file, optional
##    :type file: :class:`PathLike`

    :param string: If given, parse this string, optional
    :type string: :class:`str`
    '''
    def __init__(
        self,
        lexer: Lexer,
        # string: str,
        # file: PathLike = None,
        # *,
        # string: str = None,
    ) -> None:
##        if string is None:
##            if file is None:
##                msg = "Either a string or a filename is required"
##                raise ValueError(msg)

        # self._file = file
        # self._string = string
        # if lexer is None:
            # lexer = Lexer(string)
        self._lexer = lexer
        self._token_stack = LifoQueue()
        self._production_stack = LifoQueue()
        self._lookahead = None
        self._tokens = self._lexer.lex()
        self._cursor = 0
##      REMOVE:
        self._current_expr = None










































































































































































































































































































































































    def _parse_literal(self, tok) -> Constant:
        '''
        Parse a literal constant value at the current token::

            42  # -> Constant(42)

        :rtype: :class:`Constant`
        '''
        self._current_expr = Constant
        match tok.kind:
            # Return literals as Constants:
            case T_Literal.STRING:
                value = tok.value
            case T_Literal.INTEGER:
                value = int(tok.value)
            case T_Literal.FLOAT:
                value = float(tok.value)
            case T_Literal.TRUE:
                value = True
            case T_Literal.FALSE:
                value = False
            case _:
                raise ParseError("Expected a literal, but got " + repr(tok))
        node = Constant(value)
        print("from literal:", ast.dump(node))
        return node
        return Constant(value)

    def _parse_stmt(self) -> Assign:
        '''
        Parse an assignment statement at the current token::

            foo = ['a', 'b']  # -> Assign(
                [Name('foo')],
                List([Constant('a'), Constant('b')])
            )

        :rtype: :class:`Assign`
        '''
        while self._not_eof():
            tok = self._cur_tok()
            kind = tok.kind
            assign = self._parse_root_assign()
            print("from stmt:", ast.dump(assign))
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


    End = EndOfFile.EOF

    class Expr:
        '''
        Expr : LITERAL
             | List
             | Dict
             | Delimiter
             | None
        '''

    class RepeatedExpr:
        '''
        RepeatedExpr : Expr Delimiter RepeatedExpr
                     | None
        '''

    class RepeatedKVP:
        '''
        RepeatedKVP : KVPair Delimiter RepeatedKVP
                    | None
        '''

    class KVPair:
        '''
        KVPair : IDENTIFIER MaybeEQ Expr
        '''

    class MaybeEQ:
        '''
        MaybeEQ : EQUALS | None
        '''

    class Delimiter:
        '''
        Delimiter : NEWLINE | COMMA | None
        '''

    _tokens = (
        EndOfFile.EOF,
        Newline.NEWLINE,

        T_Literal.FLOAT,
        T_Literal.INTEGER,
        T_Literal.STRING,
        T_Literal.TRUE,
        T_Literal.FALSE,

        # T_Ignore.COMMENT,
        # T_Ignore.SPACE,

        T_Symbol.IDENTIFIER,

        T_BinOp.ASSIGN,
        T_BinOp.ATTRIBUTE,
        T_BinOp.ADD,
        T_BinOp.SUBTRACT,

        T_Syntax.COMMA,
        T_Syntax.L_PAREN,
        T_Syntax.R_PAREN,
        T_Syntax.L_BRACKET,
        T_Syntax.R_BRACKET,
        T_Syntax.L_CURLY_BRACE,
        T_Syntax.R_CURLY_BRACE,
    )

    _terminal_syms = (
        Name,
    )

    _unary_syms = ()

    _binop_syms = (
        Assign,
        Attribute,
    )

    _equivalencies = (
        (Name, T_Symbol.IDENTIFIER),
        (Assign, T_BinOp.ASSIGN),
        *((Constant, kind) for kind in T_Literal),
    )

    _terminals = tuple(
        set(_tokens)
        - {  # Remove unused tokens
            T_BinOp.ADD,
            T_BinOp.SUBTRACT,
            T_Syntax.L_PAREN,
            T_Syntax.R_PAREN
        }
    )

    def __init__(self, lexer: Lexer) -> None:
        self._lexer = lexer
        self._tokens = self._lexer.lex()
        self._cursor = 0
        self._lookahead = None

        MaybeEQ = self.MaybeEQ
        Delimiter = self.Delimiter
        Expr = self.Expr
        RepeatedExpr = self.RepeatedExpr
        KVPair = self.KVPair
        RepeatedKVP = self.RepeatedKVP

        EOF = EndOfFile.EOF
        NL = Newline.NEWLINE
        COMMA = T_Syntax.COMMA
        ID = T_Symbol.IDENTIFIER
        LB = T_Syntax.L_BRACKET
        RB = T_Syntax.R_BRACKET
        LCB = T_Syntax.L_CURLY_BRACE
        RCB = T_Syntax.R_CURLY_BRACE

        nonterms = (Assign, MaybeEQ, Delimiter, Expr, RepeatedExpr, KVPair,
                    RepeatedKVP, List, Dict)

        self._parsing_table = {

            Module: {
                EOF: None,
                ID: [(Assign, Delimiter, Module)],
            },

            Assign: {
                EOF: None,
                NL: None,
                COMMA: None,
                ID: [(Name, MaybeEQ, Expr)],
            },

            Delimiter: dict.fromkeys(self._terminals),

            MaybeEQ: dict.fromkeys(self._terminals),

            Expr: {
                EOF: Constant(None),
                NL: Constant(None),
                COMMA: Constant(None),
                LB: List,
                RB: Constant(None),
                LCB: Dict,
                RCB: Constant(None),

                T_Literal.INTEGER: Constant,
                T_Literal.FLOAT: Constant,
                T_Literal.STRING: Constant,
                T_Literal.TRUE: Constant,
                T_Literal.FALSE: Constant,
                T_Literal.NONE: Constant,
##
##                T_Literal.INTEGER: self._parse_T_Literal,
##                T_Literal.FLOAT: self._parse_T_Literal,
##                T_Literal.STRING: self._parse_T_Literal,
##                T_Literal.TRUE: self._parse_T_Literal,
##                T_Literal.FALSE: self._parse_T_Literal,
##                T_Literal.NONE: self._parse_T_Literal,
##

            },

            # Constant: {
            # },

            KVPair: {
                ID: [(Name, MaybeEQ, Expr)]
            },

            RepeatedExpr: dict.fromkeys(
                self._terminals, [(Expr, Delimiter, RepeatedExpr)]
            ),

            RepeatedKVP: {
                ID: Name,
                RCB: RCB,  #?????
            },

            List: {LB: RepeatedExpr},

            Dict: {LCB: RepeatedKVP},

        }

        table = self._parsing_table

##        for kind in T_Literal:
##            # table[Expr][kind] = [(Constant,)]
##            table[Expr][kind] = self._parse_T_Literal
##            del table[KVPair][kind]
        table[MaybeEQ][T_BinOp.ASSIGN] = T_BinOp.ASSIGN
        del table[MaybeEQ][ID]
        del table[Delimiter][T_BinOp.ASSIGN]
        # del table[KVPair][T_BinOp.ASSIGN]
        table[RepeatedExpr][T_Syntax.R_BRACKET] = None
        for k in ID, T_BinOp.ASSIGN, RCB:
            del table[RepeatedExpr][k]


    def _parse(self) -> AST:
        stack = [EndOfFile.EOF, Module]
        self._parse_stack = stack
        eval_stack = []
        self._eval_stack = eval_stack
        lookahead = None
        self._lookahead = lookahead
        get_token = self._lexer.get_token
        table = self._parsing_table

        lookahead = get_token()
        while True:
            if lookahead.kind in T_Ignore:
                lookahead = get_token()
                continue

            print()
            print(f"{stack = }")
            print(f"{lookahead = }")

            if isinstance(stack[-1], Token):
                stack_top = stack[-1].kind
                print("uh oh:", stack[-1])
            elif isinstance(stack[-1], AST):
                stack_top = type(stack[-1])
            else:
                stack_top = stack[-1]

            if stack_top in self._terminal_syms:
                typ = stack.pop()
                node = typ(lookahead.value)
                stack[-1] = node
                if not isinstance(node, Name):
                    print(node.value)
                # eval_stack.append(node)
                # stack.append(node)
                print(stack)
                lookahead = get_token()
                continue
                # return

            elif stack_top in self._unary_syms:
                pass

##            elif eval_stack:
##            # elif stack[-1] in self._binop_syms:
##                # node = stack.pop()(lookahead.value)
##                eval_stack.append(node)
##                print(stack)
##                lookahead = get_token()

            if lookahead.kind is stack_top:
                print("Exchanging!")
                # stack.pop()
                stack[-1] = lookahead
                lookahead = get_token()
                # action = table[stack_top][lookahead.kind]
                # print(f"{action = }")
                continue
            # print("Here")

            action = table[stack_top][lookahead.kind]
            print(f"{action = }")
            if action is None:
                lookahead = get_token()
                continue

            # stack.pop()
            if callable(action):
                print("Reduce!")
                print(f"{lookahead = }")
                self._lookahead = lookahead
                parsed = action()
                print(f"{parsed = }")
                # stack.pop()
                if lookahead.kind is stack_top:
                    stack[-1] = parsed
                else:
                    stack.append(action)
                    # print(f"No reduce: {stack = }, {lookahead = }")
                    raise ValueError()

            else:
                for production in action:
                        
                    if production is None:
                        lookahead = get_token()
                        continue
                    # stack.pop()
                    stack.extend(reversed(production))
                    print()
                # lookahead = get_token()

                # try:
                    # match stack element against reduction step?


            if lookahead.kind is EndOfFile.EOF:
                return

            
        '''
        a = 13
        [], T_Symbol.IDENTIFIER, '= 13'  # parse_T_Symbol()
        [Name(a)], T_BinOp.ASSIGN, '13'  # Shift, adding ASSIGN to token_stack?
        
        '''


            

    def _parse_T_Literal(self) -> Constant:
        '''
        Parse a literal constant value::

            42  # -> Constant(42)

        :rtype: :class:`Constant`
        '''
        # Literals are terminal; no further reduction is needed.
        E = self._lookahead
        match terminal.kind:
            # Return literals as Constants:
            case T_Literal.STRING:
                value = E.value
            case T_Literal.INTEGER:
                value = int(E.value)
            case T_Literal.FLOAT:
                value = float(E.value)
            case T_Literal.TRUE:
                value = True
            case T_Literal.FALSE:
                value = False
            case _:
                raise ParseError("Expected a literal, but got " + repr(E))
        node = Constant(value)
        return node

    def _parse_Attribute(self) -> Attribute:
        '''
        Parse an attribute at the current token inside an object::

            a.b.c  # -> Attribute(Attribute(Name('a'), 'b'), 'c')

        :param L: The base of the attribute to come, either a single
            variable name or a whole attribute expression
        :type L: :class:`Name` | :class:`Attribute`

        :param R: The attribute name being accessed on the base
        :type R: :class:`str`

        :rtype: :class:`Attribute`
        '''
        pass

    def _parse_T_BinOp(self) -> Name:
        while (nxt := self._lexer.get_token()).kind in T_Ignore:
            continue

        self._parse_stack.append(nxt)
        tok = self._lookahead


class FileParser:
    pass



if __name__ == '__main__':
    string = 'c = "a"'
    result = LLParser(Lexer(string))._parse()
    print(result)

