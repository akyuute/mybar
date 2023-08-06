import ast
import re
import string
import sys
from ast import (
    AST,
    Assign,
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

from ._types import FileContents

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

# AssignEvalNone = Enum(
    # 'AssignEvalNone',
    # ((k.name, k.value) for k in (*Newline, *Eof, *Unknown, )#Syntax.COMMA)
# ))
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
                text = first_tok.match.group()
            else:
                text = first_tok.value
            # highlight = '^' * len(text)
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
                with_dups = tuple(lexer.get_line(t) for t in between if t.kind not in Ignore)
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
                    continue
                    highlight += " " * len(t.value)
                    continue

                case t.kind if t.kind in Ignore:
                    continue

                case Newline.NEWLINE:
                    highlight += line_bridge
                    continue

                case Literal.STRING:
                    # match.group() has the quotation marks:
                    token_length = len(t.match.group())

                case _:
                    token_length = len(t.value)

            highlight += '^' * token_length


        # Determine how far along the first token is in the line:
        line_start_distance = len(line) - len(line.lstrip())
        tok_start_distance = first_tok.colno - line_start_distance - 1
        offset = ' ' * tok_start_distance
        highlight = dent + offset + highlight
        line = dent + line.strip()

        errmsg = leader + break_line + msg + '\n'.join(('', line, highlight))
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
        ignore = ('cursor', 'lineno', 'colno', 'lexer', 'file')
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

    def __class_getitem__(cls, item: TokenKind) -> str:
        item_name = (
            item.__class__.__name__
            if not hasattr(item, '__name__')
            else item.__name__
        )
        return f"{cls.__name__}[{item_name}]"

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
                raise TokenError.hl_error(self, msg)

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
        (re.compile(r'^' + NEWLINE + r'+'), Newline.NEWLINE),
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
        self._stream = StringIO(self._string)
        self._token_stack = LifoQueue()
        self._tokens = []

        self._cursor = 0  # 0-indexed
        self._lineno = 1  # 1-indexed
        self._colno = 1  # 1-indexed
        self.eof = EndOfFile.EOF

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
        self._lineno = 1
        self._colno = 1
        self._tokens.clear()
        return self

    def get_prev(self, back: int = 1, since: Token = None) -> tuple[Token]:
        '''
        '''
        if since is None:
            return self._tokens[-back:]
            # token = self._tokens[-1]
        # tokens = self.reset().lex()
        tokens = self._tokens
        idx = tuple(tok.cursor for tok in tokens).index(since.cursor)
        # idx = cursors.index(token.cursor)
        ret = tokens[idx - back : idx + 1]
        return ret

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
            if kind is Newline.NEWLINE:
                self._lineno += len(tok.value)
                self._colno = 1

            if kind is Literal.STRING:
                # Process strings by removing quotes:
                speech_char = tok.match.groups()[0]
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
                    match=None,
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
                match=None,
                lexer=self,
                file=self._file
            )
            try:
                if bad_token in self.SPEECH_CHARS:
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
                import traceback
                traceback.print_exc(limit=1)
                # OK, yeet:
                raise StackTraceSilencer(1)  # Sorry...


class Dict(_Dict):
    '''
    A custom ast.Dict class with instance attribute `_root`
    logging whether or not a mapping encloses other mappings.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._root = False

    def __repr__(self) -> str:
        cls = type(self).__name__
        nest = self._root
        keys = [k if isinstance(k, str) else k.id for k in self.keys]
        r = f"{cls}({nest=}, {keys=})"
        return r


class Interpreter(NodeVisitor):
    '''
    Convert ASTs to literal Python code.
    '''
    def visit_Constant(self, node: AST) -> str:
        return node.value

    def visit_Dict(self, node: AST) -> dict:
        new_d = {}

        for key, value in zip(node.keys, node.values):
            k = self.visit(key)
            v = self.visit(value)
            if key not in new_d:
                new_d[k] = v
            else:
                new_d[k].update(v)

        return new_d

    def visit_List(self, node: AST) -> list:
        print(ast.dump(node))
        return [self.visit(e) for e in node.elts]

    def visit_Name(self, node: AST) -> str:
        return node.id


class ConfigFileMaker(NodeVisitor):
    '''
    Convert ASTs to string representations with config file syntax.
    '''
    def __init__(self) -> None:
        self.indent = 4 * ' '

    def stringify(self, tree: list[AST], sep: str = '\n\n') -> FileContents:
        strings = [self.visit(n) for n in tree]
        return sep.join(strings)

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, str):
            return repr(val)
        if val is None:
            return ""
        return str(val)

    def visit_Dict(self, node):
        if node._root:
            target = self.visit(*node.keys)  # There will only be one.
            joined = self.visit(*node.values)  # There will only be one.
            string = f"{target} {{\n{self.indent}{joined}\n}}"
            return string

        assignments = []
        for k, v in zip(node.keys, node.values):
            key = self.visit(k)
            if isinstance(v, Dict):
                attrs = (self.visit(attr) for attr in v.keys)
                vals = (self.visit(value) for value in v.values)
                for attr, val in zip(attrs, vals):
                    assignments.append(f"{key}.{attr} = {val}")

            else:
                val = self.visit(v)
                assignments.append(f"{key} = {val}")

        joined = ('\n' + self.indent).join(assignments)
        return joined

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


class DictConverter:
    '''
    Convert Python dictionaries to ASTs.
    '''
    @classmethod
    def unparse(cls, mapping: dict) -> list[AST]:
        if not isinstance(mapping, dict):
            return mapping

        assignments = []

        for key, val in mapping.items():
            node = Dict([Name(key)], [cls.unparse_node(val)])
            if any(isinstance(v, Dict) for v in node.values):
                node._root = True
            assignments.append(node)
        return assignments

    @classmethod
    def unparse_node(cls, thing) -> AST:
        '''
        '''
        if isinstance(thing, dict):
            keys = []
            vals = []

            for key, val in thing.items():
                
                if key not in keys:
                    keys.append(key)
                    vals.append(cls.unparse_node(val))
                else:
                    # The equivalent of dict.update():
                    idx = keys.index(key)
                    vals[idx].keys.append(*v.keys)
                    vals[idx].values.append(*v.values)

            node = Dict([Name(k) for k in keys], vals)
            return node

        elif isinstance(thing, list):
            values = [Name(v) for v in thing]
            return List(values)

        elif isinstance(thing, (int, float, str)):
            return Constant(thing)

        elif thing in (None, True, False):
            return Constant(thing)

        return node

    @classmethod
    def as_file(cls, mapping: dict) -> FileContents:
        '''
        '''
        tree = cls.unparse(mapping)
        text = ConfigFileMaker().stringify(tree)
        return text

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
        self._tokens = self._lexer.lex()
        self._cursor = 0
        # self._previous_token = None

    def tokens(self) -> list[Token]:
        return self._lexer.lex()

    def token_dict(self) -> dict[int, Token]:
        return {t.cursor: t for t in self._lexer.lex()}

    def as_dict(self, tree = None) -> dict:
        '''
        Parse a config file and return its data as a :class:`dict`.
        '''
        if tree is None:
            self._lexer.reset()
            tree = self.parse()

        i = Interpreter()
        mapping = {}
        for n in tree:
            if not isinstance(n, Dict):
                # Each statement must map one thing to another.
                # This statement breaks the rules.
                note = (
                    f"{n._token.kind.value} cannot be at"
                    f" the start of a statement"
                )
                msg = f"Invalid syntax: {n._token.match.group()!r} ({note})"
                raise ParseError.hl_error(n._token, msg)

            parsed = i.visit(n)
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
                parsed = self.parse_stmt()

                if parsed is self._lexer.eof:
                    break

                if isinstance(parsed, Dict):
                    if any(isinstance(v, Dict) for v in parsed.values):
                        parsed._root = True

                body.append(parsed)

        except TokenError as e:
            import __main__ as possibly_repl
            if not hasattr(possibly_repl, '__file__'):
                # User is in a REPL! Don't yeet them.
                raise
            import traceback, sys
            traceback.print_exc(limit=1)
            sys.exit(1)

        self._lexer.reset()
        return body

    def parse_stmt(self, prev: Token | AST = None) -> AST:
        '''
        Parse a regular statement and return its AST.
        Each statement should result in a key-value pair represented by a
        :class:`Dict` node.
        '''
        curr = self._lexer.get_token()

        if prev is None:
            # The first token in a statement.
            if curr.kind is Syntax.COMMA:
                msg = "Invalid syntax: Comma used outside [] or {}:"
                raise ParseError.hl_error(curr, msg)

            if curr.kind in Syntax:
                msg = f"Unmatched {curr.value!r}:"
                raise ParseError.hl_error(curr, msg)

            if curr.kind in BinOp:
                note = f"expected identifier, got {curr.value!r}"
                msg = f"Invalid syntax: {note}:"
                raise ParseError.hl_error(curr, msg)

            if curr.kind in Literal:
                typ = curr.kind.value
                note = f"expected identifier, got {typ} value"
                msg = f"Invalid syntax: {note}:"
                raise ParseError.hl_error(curr, msg)

            # Advance to the next statement:
            return self.parse_stmt(prev=curr)

        if curr.kind in Literal:
            # Ensure that a literal does not begin a statement:
            if isinstance(prev, Token):
                if prev.kind not in (BinOp.ASSIGN, *Syntax):
                    typ = curr.kind.value
                    note = f"expected identifier, got {typ} value"
                    msg = f"Invalid syntax: {note}:"
                    raise ParseError.hl_error(curr, msg)

        match curr.kind:

            case self._lexer.eof:
                if prev.value in tuple('[{('):
                    msg = f"Unterminated {prev.value!r}"
                    raise ParseError.hl_error(prev, msg)

                return self._lexer.eof

            case Ignore.SPACE | Ignore.COMMENT:
                # Skip spaces and comments.
                node = self.parse_stmt(prev=prev)
                return node

            # case AssignEvalNone.NEWLINE:
            case Newline.NEWLINE:
                if prev.kind is BinOp.ASSIGN:
                    # Notify of empty assignment, if applicable.
                    node = curr
                    return node
                else:
                    # Skip, otherwise.
                    node = self.parse_stmt(prev=prev)
                    return node

            case Syntax.COMMA:
                # Continue:
                return self.parse_stmt(prev=prev)

            case Symbol.IDENTIFIER:
                if isinstance(prev, Token):
                    if prev.kind is Symbol.IDENTIFIER:
                        note = (
                            f"Multiple identifiers ({prev.value!r}"
                            f" and {curr.value!r}) not allowed together"
                        )
                        msg = f"Invalid syntax: {note}:"
                        raise ParseError.hl_error((prev, curr), msg, False)

                    # Remove once scoped self-reference is available:
                    if prev.kind is BinOp.ASSIGN:
                        # note = f"value of {curr.value!r} cannot be a variable"
                        note = f"expected literal value, got {curr.kind.value}"
                        msg = f"Invalid assignment: {note}:"
                        raise ParseError.hl_error(curr, msg)

                    if prev.kind is Syntax.L_BRACKET:
                        # Consecutive identifiers are valid inside lists.
                        node = Name(curr.value)

                    else:
                        # Continue, giving context.
                        nxt = self.parse_stmt(prev=curr)
                        node = nxt

                else:
                    # Continue, giving context.
                    nxt = self.parse_stmt(prev=curr)
                    node = nxt

            case BinOp.ASSIGN:
                if isinstance(prev, Token):

                    # Only ever assign to identifiers:

                    if prev.kind in Literal:
                        typ = prev.kind.value.lower()
                        # note = f"cannot assign to {typ} value"
                        note = f"cannot assign to literal {typ} value"
                        note += " (expected identifier)"
                        msg = f"Invalid syntax: {note}:"
                        raise ParseError.hl_error((prev, curr), msg)

                    if prev.kind is not Symbol.IDENTIFIER:
                        # This assignment is missing a target.
                        msg = f"Invalid syntax:"
                        # typ = prev.kind.value.lower()
                        # if prev.kind in Syntax:

                        if prev.kind is Syntax.L_CURLY_BRACE:
                            note = (
                                f"If this is an assignment,"
                                f" it needs a valid variable name"
                            )
                            msg = f"Invalid syntax: {note}"
                            raise ParseError.hl_error((prev, curr), msg, False)
                        elif prev.kind is Syntax.L_BRACKET:
                            note = "Assignments are not valid inside []."

                        msg = f"Invalid syntax: {note}"
                        # note = f"{curr.value!r} missing target identifier"
                        # raise ParseError.hl_error((prev, curr), msg, False)
                        raise ParseError.hl_error(curr, msg)


                target = Name(prev.value)

                val = self.parse_stmt(prev=curr)
                # if val is AssignEvalNone.EOF:
                if val is EndOfFile.EOF:
                    value = Constant(None)
                elif isinstance(val, Token):
                    # Handle empty assignments:
                    if val.kind in AssignEvalNone:
                        value = Constant(None)

                    # Unreachable?
                    else:
                        msg = f"Invalid syntax for assignment: {val.value!r}"
                        raise ParseError.hl_error((prev, curr, val), msg)

                else:
                    value = val

                node = Dict([target], [value])
                if isinstance(prev, Token) and prev.kind is Syntax.L_CURLY_BRACE:
                    # Label this node as being at root level.
                    node._root = True

            case BinOp.ATTRIBUTE:
                curr.check_expected_after(prev, (Symbol.IDENTIFIER,))
                base = Name(prev.value)
                attr = self.parse_stmt(prev=curr)
                node = Dict(keys=[base], values=[attr])

            case Syntax.L_CURLY_BRACE:
                # if prev.kind is Symbol.IDENTIFIER:
                    # self._current_assignment = prev

                keys = []
                vals = []

                # Gather assignments, which come in the form of `Dict`:
                while True:
                    assign = self.parse_stmt(prev=curr)
                    if isinstance(assign, Token):
                        if assign.kind is Syntax.R_CURLY_BRACE:
                            break

                    elif isinstance(assign, Constant):
                        typ = assign._token.kind.value
                        note = f"cannot assign to literal {typ} value"
                        note += " (expected identifier)"
                        msg = f"Invalid syntax: {note}:"
                        raise ParseError.hl_error((assign._token, ), msg)

                    if isinstance(assign, Dict):
                        if not assign.keys:
                            continue
                        
                        # Assignments only have one key and one value:
                        key = assign.keys[-1].id
                        val = assign.values[-1]

                        if key not in keys:
                            keys.append(key)
                            vals.append(val)

                        else:
                            # Make nested dicts.
                            idx = keys.index(key)
                            # Equivalent to dict.update():
                            vals[idx].keys.append(val.keys[-1])
                            vals[idx].values.append(val.values[-1])

                # Put all the assignments together:
                reconciled = Dict([Name(k) for k in keys], vals)

                # Is the enclosing assignment made using '='?
                if prev.kind is Symbol.IDENTIFIER:
                    # No. ('foo {bar=1, baz=2, ...')
                    # Wrap ourselves in Dict as if assigning to 'foo':
                    node = Dict(keys=[Name(prev.value)], values=[reconciled])
                    # Label this node as being at root level.
                    node._root = True
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

                # if prev.kind is Symbol.IDENTIFIER:
                    # self._current_assignment = prev

                b = Syntax.R_BRACKET
                elems = []
                # Gather elements.
                while True:
                    node = self.parse_stmt(prev=curr)
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
                    # Return ourself to the '=':
                    node = reconciled

            case Syntax.R_BRACKET:
                curr.check_expected_after(
                    prev, (Syntax.L_BRACKET, Syntax.COMMA)
                )
                node = curr
                # self._current_assignment = None

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




    def currtok(self) -> Token:
        return self._tokens[self._cursor]

    def advance(self) -> Token:
        self._cursor += 1
        return self._curr()

    def not_eof(self) -> bool:
        return (self.currtok().kind is not EndOfFile.EOF)
    
    def parse_stmt(self) -> AST:
        while self.not_eof():
            # tok = self._lexer.get_token()
            tok = self.currtok()
            kind = tok.kind
            # print(self.parse_expr())
            match kind:
                case BinOp.ASSIGN:
                    return self.parse_root_assign(tok)
                case BinOp.ATTRIBUTE:
                    value = self.parse_attribute(tok)
                case _:
                    raise ParseError(repr(tok))

        
        return Dict([target], [value])

    def parse_expr(self) -> AST:
        tok = self._lexer.get_token()
        kind = tok.kind
        while self.not_eof():
            match kind:
                case kind if kind in Literal:
                    return self.parse_literal(tok)
                case BinOp.ATTRIBUTE:
                    return self.parse_mapping(self.parse_attribute(tok))
                # case kind if kind in AssignEvalNone:
                case kind if kind in (*Newline, Syntax.COMMA):
                    # if self._previous_token.kind is BinOp.ASSIGN:
                        # return Constant(None)
                    # return self.parse_space
                case kind if kind in Ignore:
                    continue
                case _:
                    print(tok)

    def parse_root_assign(self, )# t: Token[Symbol.IDENTIFIER]) -> AST:
        '''
        A 1-to-1 mapping in the outermost scope::
        foo = 'bar'
        '''
        self.expect_curr(Symbol.IDENTIFIER, "Not an identifier")
        target = Name(t.value)
        target._token = target
        self.expect_next(BinOp.ASSIGN, "Not '='")
        value = self.parse_expr()
        node = Assign([target], value)
        node._token = curr
        return node

    def parse_mapping(self, )# t: Token[Symbol.IDENTIFIER]) -> AST:
        '''
        A 1-to-1 mapping inside an object::
        {key = 5}
        '''
        self.expect(t, Symbol.IDENTIFIER)
        target = Name(t.value)
        target._token = target
        self.expect_next(BinOp.ASSIGN)
        value = self.parse_expr()
        node = Dict([target], [value])
        node._token = curr
        return node


    def parse_object(self) -> AST:
        '''
        A dict containing many mappings::
        {
            foo = 'bar'
            baz = 42
            ...
        }
        '''
        self.expect_next(Syntax.L_CURLY_BRACE)
        keys = []
        vals = []

        while True:
            tok = self.advance()
            kind = tok.kind
            match kind:
                case Syntax.R_CURLY_BRACE:
                    break
                case kind if kind in Ignore:
                    continue
                case Symbol.IDENTIFIER:
                    key = tok
                    val = self.parse_mapping(key)

                    if key not in keys:
                        keys.append(key)
                        vals.append(val)

                    else:
                        # Make nested dicts.
                        idx = keys.index(key)
                        # Equivalent to dict.update():
                        vals[idx].keys.append(val.keys[-1])
                        vals[idx].values.append(val.values[-1])

                case _:
                    raise ParseError("What? " + repr(tok))

        # Put all the assignments together:
        reconciled = Dict([Name(k) for k in keys], vals)
        return reconciled

    def parse_literal(self, tok) -> AST:
        match tok.type:
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
                raise ParseError("Not a literal: " + repr(tok))
        return Constant(value)


