from collections import namedtuple
from enum import Enum, StrEnum, unique
from functools import lru_cache
from typing import TypeAlias

from .utils import (
    isalnum,
    isletter,
    isnamechar,
)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: Lexer
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
@unique
class TokenKind(StrEnum):
    TupleStart = "<<"
    TupleEnd = ">>"
    Except = "]_"
    Assign = "<-"
    Arrow = "->"
    Mapsto = "|->"
    BindTo = "=="
    ForAll = "\\A"
    Exists = "\\E"
    ForAllUnique = "\\AA"
    ExistsUnique = "\\EE"
    WeakFairness = "WF_"
    StrongFairness = "SF_"

    @classmethod
    @lru_cache
    def token_list(cls) -> list[str]:
        return [kind.value for kind in cls]

    @classmethod
    @lru_cache
    def token_set(cls) -> set[str]:
        return set(kind.value[0] for kind in cls)


Precedence: TypeAlias = tuple[int, int]
TokenPrecedence: TypeAlias = tuple[str, Precedence]

# For error messages
shortest = []
error = []


def parse_error(a, r: "Token"):
    global shortest, error
    if len(r) < len(shortest):
        error = a
        shortest = r
    return (False, a, r)


class TokenPrecedenceMixin:
    """For an enum class with TokenPrecedence values"""
    @classmethod
    @lru_cache
    def get(cls, key: str) -> Precedence | None:
        """Return the precedence of the given token"""
        for kind in cls:
            if kind.value[0] == key:
                return kind
        return None

    @classmethod
    @lru_cache
    def token_list(cls) -> list[str]:
        """Return a list of all token charss"""
        return [kind.value[0] for kind in cls]

    @classmethod
    @lru_cache
    def token_set(cls) -> set[str]:
        return set(kind.value[0] for kind in cls)

    @property
    def lexeme(self) -> str:
        return self.value[0]

    def precedence(self) -> int:
        """
        Returns the precedence of the operator.  This is a bit of a hack.
        We use the average of the precedence range of an operator to determine
        its precedence.  We don't care about checking for conflicts.
        """
        (lo, hi) = self.value[1]
        return (lo + hi) // 2


@unique
class PrefixTokenKind(TokenPrecedenceMixin, Enum):
    Neg = ("-", (12, 12))
    NegDot = ("-.", (12, 12))
    Not = ("~", (4, 4))
    LNot = ("\\lnot", (4, 4))
    Negation = ("\\neg", (4, 4))
    Box = ("[]", (4, 15))
    Diamond = ("<>", (4, 15))
    Domain = ("DOMAIN", (9, 9))
    Enabled = ("ENABLED", (4, 15))
    Subset = ("SUBSET", (8, 8))
    Unchanged = ("UNCHANGED", (4, 15))
    Union = ("UNION", (8, 8))


@unique
class InfixTokenKind(TokenPrecedenceMixin, TokenPrecedence, Enum):
    Ampersand = ("&", (13, 13))
    And = ("/\\", (3, 3))  # This often starts an expression as well
    Backslash = ("\\", (8, 8))
    BangBang = ("!!", (9, 13))
    Bar = ("|", (10, 11))
    BarDash = ("|-", (5, 5))
    BarEqual = ("|=", (5, 5))
    Caret = ("^", (14, 14))
    CaretCaret = ("^^", (14, 14))
    ColonColonEqual = ("::=", (5, 5))
    ColonEqual = (":=", (5, 5))
    ColonGreaterThan = (":>", (7, 7))
    Cross = ("(\\X)", (13, 13))
    Dollar = ("$", (9, 13))
    DollarDollar = ("$$", (9, 13))
    Dot = (".", (17, 17))
    DotDot = ("..", (9, 9))
    DotDotDot = ("...", (9, 9))
    DoubleAmpersand = ("&&", (13, 13))
    DoubleAt = ("@@", (6, 6))
    DoubleBar = ("||", (10, 11))
    DoubleQuestion = ("??", (9, 13))
    Equal = ("=", (5, 5))
    EqualBar = ("=|", (5, 5))
    LessThanEqual = ("=<", (5, 5))
    GreaterThan = (">", (5, 5))
    GreaterThanEqual = (">=", (5, 5))
    Hash = ("#", (5, 5))
    HashHash = ("##", (9, 13))
    RightImplies = ("=>", (1, 1))
    LeadsTo = ("~>", (2, 2))
    LeftImplies = ("<=", (5, 5))
    LessThan = ("<", (5, 5))
    LessThanColon = ("<:", (7, 7))
    Minus = ("-", (11, 11))
    MinusBar = ("-|", (5, 5))
    MinusMinus = ("--", (11, 11))
    MinusPlusArrow = ("-+->", (2, 2))
    MutualImplies = ("<=>", (2, 2))
    NotEqual = ("/=", (5, 5))
    Or = ("\\/", (3, 3))   # This often starts an expression as well
    ParensDot = ("(.)", (11, 11))
    ParensMinus = ("(-)", (11, 11))
    ParensPlus = ("(+)", (10, 10))
    ParensSlash = ("(/)", (13, 13))
    Percent = ("%", (10, 11))
    PercentPercent = ("%%", (10, 11))
    Plus = ("+", (10, 10))
    PlusPlus = ("++", (10, 10))
    Slash = ("/", (13, 13))
    SlashSlash = ("//", (13, 13))
    Star = ("*", (13, 13))
    StarStar = ("**", (13, 13))
    Approx = ("\\approx", (5, 5))
    GreaterEqual = ("\\geq", (5, 5))
    OSlash = ("\\oslash", (13, 13))
    SquareSupersetEqual = ("\\sqsupseteq", (5, 5))
    Asymptotic = ("\\asymp", (5, 5))
    DoubleGreater = ("\\gg", (5, 5))
    OTimes = ("\\otimes", (13, 13))
    BacklashStar = ("\\star", (13, 13))
    BigCircle = ("\\bigcirc", (13, 13))
    In = ("\\in", (5, 5))
    NotIn = ("\\notin", (5, 5))
    Precedes = ("\\prec", (5, 5))
    Subset = ("\\subset", (5, 5))
    Bullet = ("\\bullet", (13, 13))
    Intersect = ("\\intersect", (8, 8))
    PrecedesEqual = ("\\preceq", (5, 5))
    SubsetEqual = ("\\subseteq", (5, 5))
    Cap = ("\\cap", (8, 8))
    Land = ("\\land", (3, 3))
    Proportional = ("\\propto", (5, 5))
    Succeeds = ("\\succ", (5, 5))
    CDot = ("\\cdot", (5, 14))
    LessEqual = ("\\leq", (5, 5))
    Similar = ("\\sim", (5, 5))
    SucceedsEqual = ("\\succeq", (5, 5))
    Circle = ("\\circ", (13, 13))
    DoubleLess = ("\\ll", (5, 5))
    SimilarEqual = ("\\simeq", (5, 5))
    Superset = ("\\supset", (5, 5))
    Congruent = ("\\cong", (5, 5))
    Lor = ("\\lor", (3, 3))
    SquareCap = ("\\sqcap", (9, 13))
    SupersetEqual = ("\\supseteq", (5, 5))
    Cup = ("\\cup", (8, 8))
    Concat = ("\\o", (13, 13))
    SquareCup = ("\\sqcup", (9, 13))
    Union = ("\\union", (8, 8))
    Divide = ("\\div", (13, 13))
    ODot = ("\\odot", (13, 13))
    SquareSubset = ("\\sqsubset", (5, 5))
    UPlus = ("\\uplus", (9, 13))
    DotEqual = ("\\doteq", (5, 5))
    OMinus = ("\\ominus", (11, 11))
    SquareSubsetEqual = ("\\sqsubseteq", (5, 5))
    Wr = ("\\wr", (9, 14))
    Equivalent = ("\\equiv", (2, 2))
    OPlus = ("\\oplus", (10, 10))
    SquareSuperset = ("\\sqsupset", (5, 5))
    Product = ("\\X", (10, 13))
    Times = ("\\times", (10, 13))


@unique
class PostfixTokenKind(TokenPrecedenceMixin, TokenPrecedence, Enum):
    Prime = ("'", (15, 15))
    CaretPlus = ("^+", (15, 15))
    CaretStar = ("^*", (15, 15))
    CaretHash = ("^#", (15, 15))
    Bracket = ("[", (16, 16))


# Only tokens matching the predicate are added to lexer search
def token_predicate(op):
    return not isalnum(op[0]) and not (
        len(op) > 1 and op[0] == "\\" and isletter(op[1])
    )


def build_token_list() -> list[str]:
    """Build a list of all tokens"""
    return [
        *TokenKind.token_list(),
        *PrefixTokenKind.token_list(),
        *InfixTokenKind.token_list(),
        *PostfixTokenKind.token_list(),
    ]


class Token:
    """Data structure for a token: we expect to create a lot of these"""
    __slots__ = ("lexeme", "where", "column", "first")

    def __init__(self, lexeme: str, where: tuple[str, int], column: int, first):
        self.lexeme = lexeme  # chars representation of the token
        self.where = where  # filename, line number
        self.column = column  # Column number offset
        self.first = first  # True if it's the first token on the line

    def __getitem__(self, i):
        """Pretend to be a tuple"""
        if i == 0 or i == "lexeme":
            return self.lexeme
        if i == 1 or i == "where":
            return self.where
        if i == 2 or i == "column":
            return self.column
        if i == 3 or i == "first":
            return self.first
        raise IndexError

    def __str__(self):
        (file, line) = self.where
        return f"Token('{self.lexeme}', {file}: ({line=}, col={self.column}), {self.first})"

    @property
    def junct(self):
        """A disjunct or conjunct token is identifier by all but the line in the token"""
        return (self.lexeme, self.column, self.first)


KNOWN_TOKENS = list(filter(token_predicate, build_token_list()))

# Turn input into a sequence of tokens.  Each token is a tuple
#   (lexeme, (file, line), column, first), where first is true if
#   it's the first token on the line
def lexer(chars: str, filename: str) -> list[Token]:
    found_tokens: list[Token] = []
    line = 1
    column = 1
    first = True
    while chars != "":
        # see if it's a blank
        if chars[0] in {" ", "\t"}:
            chars = chars[1:]
            column += 1
            continue

        if chars[0] == "\n":
            chars = chars[1:]
            line += 1
            column = 1
            first = True
            continue

        # Skip over "pure" TLA+
        if chars.startswith("\\*++:SPEC"):
            chars = chars[8:]
            while len(chars) > 0 and not chars.startswith("\\*++:PlusPy"):
                chars = chars[1:]
            continue

        # skip over line comments
        if chars.startswith("\\*"):
            chars = chars[2:]
            while len(chars) > 0 and chars[0] != "\n":
                chars = chars[1:]
            continue

        # skip over nested comments
        if chars.startswith("(*"):
            count = 1
            chars = chars[2:]
            column += 2
            while count != 0 and chars != "":
                if chars.startswith("(*"):
                    count += 1
                    chars = chars[2:]
                    column += 2
                elif chars.startswith("*)"):
                    count -= 1
                    chars = chars[2:]
                    column += 2
                elif chars[0] == "\n":
                    chars = chars[1:]
                    line += 1
                    column = 1
                    first = True
                else:
                    chars = chars[1:]
                    column += 1
            continue

        # a series of four or more '-' characters is a lexeme
        if chars.startswith("----"):
            chars = chars[4:]
            col = column
            column += 4
            while len(chars) > 0 and chars[0] == "-":
                chars = chars[1:]
                column += 1
            found_tokens.append(Token("----", (filename, line), col, first))
            first = False
            continue

        # a series of four or more '=' characters is a lexeme
        if chars.startswith("===="):
            chars = chars[4:]
            col = column
            column += 4
            while len(chars) > 0 and chars[0] == "=":
                chars = chars[1:]
                column += 1
            found_tokens.append(Token("====", (filename, line), col, first))
            first = False
            continue

        # if a backslash, it may be an special operator.  Otherwise just \
        if chars[0] == "\\" and len(chars) > 1 and isalnum(chars[1]):
            i = 2
            while i < len(chars) and isalnum(chars[i]):
                i += 1
            found_tokens.append(Token(chars[:i], (filename, line), column, False))
            first = False
            chars = chars[i:]
            column += i
            continue

        # see if it's a multi-character token.  Match with the longest one
        found = ""
        for tok in KNOWN_TOKENS:
            if chars.startswith(tok) and len(tok) > len(found):
                found = tok
        if found != "":
            found_tokens.append(Token(found, (filename, line), column, first))
            first = False
            chars = chars[len(found) :]
            column += len(found)
            continue

        # see if a sequence of letters and numbers
        if isnamechar(chars[0]):
            i = 0
            while i < len(chars) and isnamechar(chars[i]):
                i += 1
            found_tokens.append(Token(chars[:i], (filename, line), column, first))
            first = False
            chars = chars[i:]
            column += i
            continue

        # chars
        if chars[0] == '"':
            i = 1
            str = '"'
            while i < len(chars) and chars[i] != '"':
                if chars[i] == "\\":
                    i += 1
                    if i == len(chars):
                        break
                    if chars[i] == '"':
                        str += '"'
                    elif chars[i] == "\\":
                        str += "\\"
                    elif chars[i] == "t":
                        str += "\t"
                    elif chars[i] == "n":
                        str += "\n"
                    elif chars[i] == "f":
                        str += "\f"
                    elif chars[i] == "r":
                        str += "\r"
                    else:
                        str += chars[i]
                else:
                    str += chars[i]
                i += 1
            if i < len(chars):
                i += 1
            str += '"'
            found_tokens.append(Token(str, (filename, line), column, first))
            first = False
            chars = chars[i:]
            column += i
            continue

        # everything else is a single character token
        found_tokens.append(Token(chars[0], (filename, line), column, first))
        first = False
        chars = chars[1:]
        column += 1

    # We discard the preamble tokens below.
    #
    # Preamble is defined as anything that comes before the module start
    # tokens which are `AtLeast4("-"), tok("MODULE"), Name, AtLeast4("-")` as
    # defined in [1].
    #
    # We could have forwarded them to the parser and handled them there.
    #
    # - We discard comments right here. No tokens are created for them
    # and the parser doesn't have to worry about them.
    # - Preamble is also like a comment. It is not useful in later stages.
    #
    # Discarding preamble tokens here keeps its handling consistent with
    # that of the comments and avoids complicating the parser code.
    #
    # For details see [2].
    #
    # References:
    # [1] https://lamport.azurewebsites.net/tla/TLAPlus2Grammar.tla
    # [2] https://github.com/tlaplus/PlusPy/issues/7
    while True:
        if len(found_tokens) < 4:
            break
        at_least_4_before = found_tokens[0].lexeme == "----"
        tok_module = found_tokens[1].lexeme == "MODULE"
        at_least_4_after = found_tokens[3].lexeme == "----"
        if at_least_4_before and tok_module and at_least_4_after:
            break
        else:
            # Advance the line number and discard the token
            found_tokens = found_tokens[1:]

    return found_tokens
