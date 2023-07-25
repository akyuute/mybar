import ast
import re
import string
import sys
from ast import (
    AST,
    Assign,
    Attribute,
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


# FILE = 'test_config.conf'
FILE = '/home/sam/.config/mybar/mybar.conf'
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
        # keys = [k if isinstance(k, str) else k.id for k in self.keys]
        keys = [
            k.id if isinstance(k, Name)
            # else ast.dump(k) if isinstance(k, Attribute)
            else f"{k.value}.{k.attr}" if isinstance(k, Attribute)
            else k
            for k in self.keys
        ]
        r = f"{cls}({keys=})"
        return r


class Interpreter(NodeVisitor):
    '''
    Convert ASTs to literal Python code.
    '''
    _EXPR_PLACEHOLDER = '_EXPR_PLACEHOLDER'

    def visit_Assign(self, node: AST) -> dict:
        return {self.visit((node.targets[-1])): self.visit(node.value)}

    def visit_Attribute(self, node: AST) -> dict:
        base = self.visit(node.value)
        attr = self.visit(node.attr)
        if isinstance(base, str):
            return {base: {attr: self._EXPR_PLACEHOLDER}}
        return self._process_nested_attr(base, attr)

    def _process_nested_attr(
        self,
        dct: dict,
        newattr: str,
        store: bool = False
    ) -> dict:
        '''
        '''
        # print(f"{newattr = }")
        for k, v in dct.items():
            # print(k, v)
            if isinstance(v, dict):
                if k == newattr:
                    # Don't clobber:
                    dct[k] = self._process_nested_attr(v, newattr, store)
                else:
                    dct[k].update(self._process_nested_attr(v, newattr, store))
            elif v is self._EXPR_PLACEHOLDER:
                if store:
                    dct[k] = newattr
                else:
                    dct[k] = {newattr: self._EXPR_PLACEHOLDER}
        return dct

    def visit_Constant(self, node: AST) -> str | int | float | None:
        return node.value

    def _nested_update(
        self, dct: dict | Any,
        upd: dict,
        assign: Any
    ) -> dict:
        dct = dct.copy()
        upd = upd.copy()
        if not isinstance(dct, dict):
            return {upd.popitem()[0]: assign}
        for k, v in upd.items():
            if v is self._EXPR_PLACEHOLDER:
                v = assign
            if isinstance(v, dict):
                dct[k] = self._nested_update(dct.get(k, {}), v, assign)
            else:
                dct[k] = v
        return dct

    def visit_Dict(self, node: AST) -> dict:
        new_d = {}
        keys = []
        vals = []

        for key, val in zip(node.keys, node.values):
            # print(ast.dump(key))
            target = self.visit(key)
            value = self.visit(val)
            if isinstance(target, dict):
                new_d = self._nested_update(new_d, target, value)
            else:
                new_d.update({target: value})

        return new_d

    def visit_List(self, node: AST) -> list:
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

    def visit_Attribute(self, node: AST) -> dict:
        base = self.visit(node.value)
        attr = self.visit(node.attr)
        return f"{base}.{attr}"

    def visit_Assign(self, node: AST) -> dict:
        target = self.visit(node.targets[-1])
        value = self.visit(node.value)
        return f"{target} = {value}"

    def visit_Constant(self, node):
        val = node.value
        if isinstance(val, str):
            return repr(val)
        if val is None:
            return ""
        return str(val)

    def visit_Dict(self, node):
        keys = (self.visit(k) for k in node.keys)
        values = (self.visit(v) for v in node.values)
        assignments = (' = '.join(pair) for pair in zip(keys, values))
        joined = f"\n{self.indent}".join(assignments)
        string = f"{{\n{self.indent}{joined}\n}}"
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
    Convert Python objects to ASTs.
    '''
    @classmethod
    def unparse(cls, mapping: dict) -> list[AST]:
        if not isinstance(mapping, dict):
            return mapping

        assignments = []
        for key, val in mapping.items():
            node = Assign([Name(key)], cls.unparse_node(val))
            assignments.append(node)
        return assignments

    @classmethod
    def _unparse_dict(cls, mapping: dict, root_name: str = None) -> tuple[AST]:
        '''
        '''
        for base, attr_or_assign in mapping.items():
            if isinstance(attr_or_assign, dict):
                attr, assign = cls._unparse_dict(attr_or_assign)
                if root_name is None:
                    name = Attribute(Name(base), attr)
                else:
                    root = Attribute(Name(root_name), Name(base))
                    name = Attribute(root, attr)
                return name, assign
            elif root_name is not None:
                root = Attribute(Name(root_name), Name(base))
                return root, attr_or_assign
            else:
                return Name(base), attr_or_assign
                # Enable updates later on

    @classmethod
    def unparse_node(cls, thing) -> AST:
        if isinstance(thing, dict):
            key_ref = []
            keys = []
            vals = []

            for key, val in thing.items():
                # print(f"{key = }  {val = }  {enclosing = }")

                if isinstance(val, dict):
                    # An attribute, possibly nested.
                # if key in key_ref:  # Handle multiple pairs later.
                    # key.val
                    name, assign = cls._unparse_dict(val, root_name=key)
                    # print(f"{ast.dump(name) = }  {assign = }")
                    keys.append(name)
                    vals.append(cls.unparse_node(assign))

                else:
                    # An assignment.
                    keys.append(Name(key))
                    vals.append(cls.unparse_node(val))

                # print(list(map(ast.dump, keys,)), list(map(ast.dump, vals)))

    ##            else:
    ##                # A root dict
    ##
    ##            # key_ref.append(name)
    ##            # key = Attribute(Name(key), cls.unparse_node(val, thing))
    ##            # keys.append(key)
    ##            # vals.append(cls.unparse_node(val, thing))
    ##
    ##                # The equivalent of dict.update():
    ##
    ##                idx = key_ref.index(key)
    ##                vals[idx].keys.extend(v)
    ##                vals[idx].values.extend(v.values())


            return Dict(keys, vals)

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
    def _nested_update(cls, dct: dict | Any, upd: dict, assign: Any) -> dict:
        if not isinstance(dct, dict):
            return {upd.popitem()[0]: assign}
        for k, v in upd.items():
            if v is self._EXPR_PLACEHOLDER:
                v = assign
            if isinstance(v, dict):
                dct[k] = cls._nested_update(dct.get(k, {}), v, assign)
            else:
                dct[k] = v
        return dct

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
        self._current_expr = None
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
            tree = self.get_stmts()
        # print(tree)

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
                msg = f"Invalid syntax: {n._token.match.group()!r} ({note})"
                raise ParseError.hl_error(n._token, msg)

            parsed = i.visit(n)
            mapping.update(parsed)

        return mapping



    def cur_tok(self) -> Token:
        return self._tokens[self._cursor]

    def expect_curr(self, kind: TokenKind, errmsg: str) -> NoReturn | bool:
        tok = self.cur_tok()
        if tok.kind is not kind:
            raise ParseError.hl_error(tok, errmsg)
        return True

    def expect_next(self, kind: TokenKind, errmsg: str) -> NoReturn | bool:
        if not isinstance(kind, tuple):
            kind = (kind,)
        tok = self.advance()
        # print(tok)
        if tok.kind not in kind:
            raise ParseError.hl_error(tok, errmsg)
        return tok

    def advance(self) -> Token:
        # if self.not_eof():
        if self._cursor < len(self._tokens) - 1:
            self._cursor += 1
        # else:
            # print(self._tokens[-1])
        curr = self.cur_tok()
        return curr
        # return self._tokens[-1]

    def reset(self) -> None:
        self._cursor = 0

    def not_eof(self) -> bool:
        return (self.cur_tok().kind is not EndOfFile.EOF)

    def is_empty(self) -> bool:
        # print(self.cur_tok().kind)
        return (self.cur_tok().kind in (*Ignore, Newline.NEWLINE))
    
        return Dict([target], [value])


    def parse_root_assign(self, ) :# t: Token[Symbol.IDENTIFIER]) -> AST:
        '''
        A 1-to-1 mapping in the outermost scope::
        foo = 'bar'
        '''
        ##print("Parsing statement")
        self.expect_curr(Symbol.IDENTIFIER, "Not an identifier")
        self._current_expr = Assign
        curr = self.cur_tok()
        target = Name(curr.value)
        # print(f"Assigning to {ast.dump(target)}")
        target._token = curr
        possible_starts = (BinOp.ASSIGN, Syntax.L_CURLY_BRACE)
        curr = self.expect_next(possible_starts, "Not '=' or '{'")
        if curr.kind is Syntax.L_CURLY_BRACE:
            value = self.parse_object()
        else:
            value = self.parse_expr()
        # if value is None:
            # value = Constant(None)
        node = Assign([target], value)
        node._token = curr
        # print(ast.dump(node))
        self.advance()

        # Needed?:
        while self.is_empty():
            # print("advance:", self.advance())
            self.advance()
        # while self.not_eof():
            # if self.is_empty():
                # print("advance:", self.advance())
                # self.advance()
            # if self.advance().kind is EndOfFile.EOF:
                # break
        return node


    def parse_expr(self) -> AST:
        ##print("Starting parse_expr()")
        # tok = self._lexer.get_token()
        while self.not_eof():
            tok = self.advance()
            # print(tok)
            kind = tok.kind

            match kind:
                case EndOfFile.EOF:
                    return kind
                    # return tok
                case kind if kind in Ignore:
                    # print("Ignoring")
                    continue
                # case Syntax.COMMA:
                    # return tok
                case Newline.NEWLINE | Syntax.COMMA:
                    ##print(tok, self._current_expr)
                    if self._current_expr in (Assign, Dict):
                        node = Constant(None)
                    else:
                        continue
                    # return tok
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
                    ##print(tok)
                    ##print("Huh?")
                    return tok
                    # node = tok

            # self.advance()
            node._token = tok
            ##print("returning", node)
            return node

    def parse_list(self) -> List:
        ##print("Parsing list")
        self.expect_curr(Syntax.L_BRACKET, "")
        self._current_expr = List
        elems = []
        while True:
            elem = self.parse_expr()
            if isinstance(elem, Token):
                if elem.kind is Syntax.R_BRACKET:
                    break
            elems.append(elem)
        return List(elems)

        
    def parse_mapping(self, ) :# t: Token[Symbol.IDENTIFIER]) -> AST:
        '''
        A 1-to-1 mapping inside an object::
        {key = 5}
        '''
        ##print("Parsing mapping")
        self.expect_curr(Symbol.IDENTIFIER, "Not an identifier")
        self._current_expr = Dict
        target_tok = self.cur_tok()
        target = Name(target_tok.value)
        target._token = target_tok

        ##print(f"{target_tok = }")

        maybe_attr = self.advance()
        ##print(f"{maybe_attr = }")
        if maybe_attr.kind is BinOp.ATTRIBUTE:
            target = self.parse_attribute(target)

        assign_tok = self.expect_curr(BinOp.ASSIGN, "Not '=' or '.'")
        value = self.parse_expr()
        # Disallow assigning identifiers:
        if isinstance(value, Name):
            typ = value._token.kind.value
            note = f"expected expression, got {typ}"
            msg = f"Invalid assignment: {note}:"
            raise ParseError.hl_error(value._token, msg)
        # if isinstance(value, Token):
            # if value.kind in (Newline.NEWLINE, Syntax.COMMA):
                # value = Constant(None)
        node = Dict([target], [value])
        node._token = assign_tok
        ##print(f"{ast.dump(node) = }")
        return node

    def parse_attribute(self, value: Name | Attribute) -> Attribute:
        ##print("parsing attribute")
        # base_tok = self.cur_tok()
        curr = self.cur_tok()
        # curr = self.advance()
        msg = "parse_attr: Not an identifier"
        # while curr.kind is BinOp.ATTRIBUTE:
        # while True:
        curr = self.expect_curr(BinOp.ATTRIBUTE, "parse_attribute() called when no '.' found")
        maybe_attr = self.expect_next(Symbol.IDENTIFIER, msg)
            # value = Name(base_tok)
            # value._token = base_tok

        attr = Name(maybe_attr.value)
        attr._token = maybe_attr

        target = Attribute(value, attr)
        target._token = curr
        # base_tok = self.expect_next(Symbol.IDENTIFIER, msg)
        # curr = self.advance()
        # if curr.kind is Symbol.IDENTIFIER:
            # return self.parse_attribute(
        maybe_another = self.advance()
        if maybe_another.kind is BinOp.ATTRIBUTE:
            target = self.parse_attribute(target)
            # target = self.parse_attribute(target)
##                target = Attribute(target, self.parse_attribute())
##            curr = self.advance()

        ##print(f"parse_attr {ast.dump(target) = }")
        return target
            


    def parse_object(self) -> AST:
        '''
        A dict containing many mappings::
        {
            foo = 'bar'
            baz = 42
            ...
        }
        '''
        ##print("Parsing obj")
        self.expect_curr(Syntax.L_CURLY_BRACE, "Not '{'")
        self._current_expr = Dict
        keys = []
        vals = []

        while True:
            tok = self.advance()
            kind = tok.kind
            match kind:
                case Syntax.R_CURLY_BRACE:
                    # print("BREAKING object")
                    break
                case kind if kind in (*Ignore, Syntax.COMMA, Newline.NEWLINE):
                    continue
                case Symbol.IDENTIFIER:
                    # key = tok.value
                    pair = self.parse_mapping()
                    key = pair.keys[-1]
                    val = pair.values[-1]

                    if key not in keys:
                        keys.append(key)
                        vals.append(val)

                    else:
                        # Make nested dicts.
                        idx = keys.index(key)
                        # Equivalent to dict.update():
                        vals[idx].keys.append(val.keys[-1])
                        vals[idx].values.append(val.values[-1])

                    # print(keys)
                case _:
                    note = f"expected variable name, got {kind.value}"
                    msg = f"Invalid target for assignment: {note}:"
                    raise ParseError.hl_error(tok, msg)

        # Put all the assignments together:
        # reconciled = Dict([Name(k) for k in keys], vals)
        reconciled = Dict(keys, vals)
        ##print("reconciled:")
        ##print(ast.dump(reconciled))
        return reconciled

    def parse_literal(self, tok) -> AST:
        ##print("Parsing literal")
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
                raise ParseError("Not a literal: " + repr(tok))
        return Constant(value)



    def parse_stmt(self) -> AST:
        while self.not_eof():
            # tok = self._lexer.get_token()
            tok = self.cur_tok()
            # print("New stmt:", tok)
            kind = tok.kind
            assign = self.parse_root_assign()
            # print(ast.dump(assign))
            # print()
            return assign
            # print(self.parse_root_assign())
            # print(self.parse_expr())
            match kind:
                case BinOp.ASSIGN:
                    return self.parse_root_assign(tok)
                case BinOp.ATTRIBUTE:
                    value = self.parse_attribute(tok)
                case _:
                    raise ParseError(repr(tok))

        
    def get_stmts(self) -> list[AST]:
        self.reset()
        # print(self._tokens)
        # for t in self._tokens:
            # print(repr(t.value))
        body = []
        while self.not_eof():
            body.append(self.parse_stmt())
            # for n in body:
                # print(ast.dump(n))
                # print()
        self.reset()
        return body



if __name__ == '__main__':
    p = Parser()
    # print(ast.dump(p.get_stmts()[0]))
    # print(ast.dump(p.get_stmts()[-1].value))
    for t in p.get_stmts():
        print(ast.dump(t))
        # print([ast.dump(t) for t in p.get_stmts()])
    d = p.as_dict()
    print()
    print(d)
    # print(d['net_stats'])
    # print()
    u = DictConverter.unparse(d)
    for t in u:
        print(ast.dump(t))
    # print([ast.dump(t) for t in u])
    # print(ast.dump(p.get_stmts()[-1].value))
    # print(ast.dump(u[-1].value))
    # print(*map(ast.dump, u))
    print()
    c = ConfigFileMaker().stringify(u)
    print(c)

