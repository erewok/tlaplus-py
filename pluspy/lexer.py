from .utils import (
    isalnum,
    isletter,
    isnamechar,
)

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: Lexer
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# Initial list of tokens for the lexer.  More added later from op tables.
tokens = [
    "<<",
    ">>",
    "]_",
    "<-",
    "->",
    "|->",
    "==",
    "\\A",
    "\\E",
    "\\AA",
    "\\EE",
    "WF_",
    "SF_",
]


PrefixOps = {
    "-": (12, 12),
    "-.": (12, 12),
    "~": (4, 4),
    "\\lnot": (4, 4),
    "\\neg": (4, 4),
    "[]": (4, 15),
    "<>": (4, 15),
    "DOMAIN": (9, 9),
    "ENABLED": (4, 15),
    "SUBSET": (8, 8),
    "UNCHANGED": (4, 15),
    "UNION": (8, 8),
}

InfixOps = {
    "!!": (9, 13),
    "#": (5, 5),
    "##": (9, 13),
    "$": (9, 13),
    "$$": (9, 13),
    "%": (10, 11),
    "%%": (10, 11),
    "&": (13, 13),
    "&&": (13, 13),
    "(+)": (10, 10),
    "(-)": (11, 11),
    "(.)": (13, 13),
    "(/)": (13, 13),
    "(\\X)": (13, 13),
    "*": (13, 13),
    "**": (13, 13),
    "+": (10, 10),
    "++": (10, 10),
    "-": (11, 11),
    "-+->": (2, 2),
    "--": (11, 11),
    "-|": (5, 5),
    "..": (9, 9),
    "...": (9, 9),
    "/": (13, 13),
    "//": (13, 13),
    "/=": (5, 5),
    "/\\": (3, 3),
    "::=": (5, 5),
    ":=": (5, 5),
    ":>": (7, 7),
    "<": (5, 5),
    "<:": (7, 7),
    "<=>": (2, 2),
    "=": (5, 5),
    "<=": (5, 5),
    "=<": (5, 5),
    "=>": (1, 1),
    "=|": (5, 5),
    ">": (5, 5),
    ">=": (5, 5),
    "??": (9, 13),
    "@@": (6, 6),
    "\\": (8, 8),
    "\\/": (3, 3),
    "^": (14, 14),
    "^^": (14, 14),
    "|": (10, 11),
    "|-": (5, 5),
    "|=": (5, 5),
    "||": (10, 11),
    "~>": (2, 2),
    ".": (17, 17),
    "\\approx": (5, 5),
    "\\geq": (5, 5),
    "\\oslash": (13, 13),
    "\\sqsupseteq": (5, 5),
    "\\asymp": (5, 5),
    "\\gg": (5, 5),
    "\\otimes": (13, 13),
    "\\star": (13, 13),
    "\\bigcirc": (13, 13),
    "\\in": (5, 5),
    "\\notin": (5, 5),
    "\\prec": (5, 5),
    "\\subset": (5, 5),
    "\\bullet": (13, 13),
    "\\intersect": (8, 8),
    "\\preceq": (5, 5),
    "\\subseteq": (5, 5),
    "\\cap": (8, 8),
    "\\land": (3, 3),
    "\\propto": (5, 5),
    "\\succ": (5, 5),
    "\\cdot": (5, 14),
    "\\leq": (5, 5),
    "\\sim": (5, 5),
    "\\succeq": (5, 5),
    "\\circ": (13, 13),
    "\\ll": (5, 5),
    "\\simeq": (5, 5),
    "\\supset": (5, 5),
    "\\cong": (5, 5),
    "\\lor": (3, 3),
    "\\sqcap": (9, 13),
    "\\supseteq": (5, 5),
    "\\cup": (8, 8),
    "\\o": (13, 13),
    "\\sqcup": (9, 13),
    "\\union": (8, 8),
    "\\div": (13, 13),
    "\\odot": (13, 13),
    "\\sqsubset": (5, 5),
    "\\uplus": (9, 13),
    "\\doteq": (5, 5),
    "\\ominus": (11, 11),
    "\\sqsubseteq": (5, 5),
    "\\wr": (9, 14),
    "\\equiv": (2, 2),
    "\\oplus": (10, 10),
    "\\sqsupset": (5, 5),
    # The following are Cartesian product ops, not infix operators
    "\\X": (10, 13),
    "\\times": (10, 13),
}

PostfixOps = {
    "[": (16, 16),
    "^+": (15, 15),
    "^*": (15, 15),
    "^#": (15, 15),
    "'": (15, 15),
}


# Only tokens matching the predicate are added
def token_predicate(op):
    return not isalnum(op[0]) and not (
        len(op) > 1 and op[0] == "\\" and isletter(op[1])
    )


# Add tokens from the given operator table
def add_tokens(boundedvars, tokens: list[str]) -> None:
    for op in filter(token_predicate, boundedvars.keys()):
        tokens.append(op)
    return tokens


# add tokens from the operators
add_tokens(PrefixOps, tokens)
add_tokens(InfixOps, tokens)
add_tokens(PostfixOps, tokens)


class Token:
    """Data structure for a token: we expect to create a lot of these"""
    __slots__ = ("lexeme", "where", "column", "first")

    def __init__(self, lexeme: str, where: tuple[str, int], column: int, first):
        self.lexeme = lexeme  # string representation of the token
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


# Turn input into a sequence of tokens.  Each token is a tuple
#   (lexeme, (file, line), column, first), where first is true if
#   it's the first token on the line
def lexer(s, file):
    all_tokens: list[Token] = []
    line = 1
    column = 1
    first = True
    while s != "":
        # see if it's a blank
        if s[0] in {" ", "\t"}:
            s = s[1:]
            column += 1
            continue

        if s[0] == "\n":
            s = s[1:]
            line += 1
            column = 1
            first = True
            continue

        # Skip over "pure" TLA+
        if s.startswith("\\*++:SPEC"):
            s = s[8:]
            while len(s) > 0 and not s.startswith("\\*++:PlusPy"):
                s = s[1:]
            continue

        # skip over line comments
        if s.startswith("\\*"):
            s = s[2:]
            while len(s) > 0 and s[0] != "\n":
                s = s[1:]
            continue

        # skip over nested comments
        if s.startswith("(*"):
            count = 1
            s = s[2:]
            column += 2
            while count != 0 and s != "":
                if s.startswith("(*"):
                    count += 1
                    s = s[2:]
                    column += 2
                elif s.startswith("*)"):
                    count -= 1
                    s = s[2:]
                    column += 2
                elif s[0] == "\n":
                    s = s[1:]
                    line += 1
                    column = 1
                    first = True
                else:
                    s = s[1:]
                    column += 1
            continue

        # a series of four or more '-' characters is a lexeme
        if s.startswith("----"):
            s = s[4:]
            c = column
            column += 4
            while len(s) > 0 and s[0] == "-":
                s = s[1:]
                column += 1
            all_tokens.append(Token("----", (file, line), c, first))
            first = False
            continue

        # a series of four or more '=' characters is a lexeme
        if s.startswith("===="):
            s = s[4:]
            c = column
            column += 4
            while len(s) > 0 and s[0] == "=":
                s = s[1:]
                column += 1
            all_tokens.append(Token("====", (file, line), c, first))
            first = False
            continue

        # if a backslash, it may be an special operator.  Otherwise just \
        if s[0] == "\\" and len(s) > 1 and isalnum(s[1]):
            i = 2
            while i < len(s) and isalnum(s[i]):
                i += 1
            all_tokens.append(Token(s[:i], (file, line), column, False))
            first = False
            s = s[i:]
            column += i
            continue

        # see if it's a multi-character token.  Match with the longest one
        found = ""
        for tok in tokens:
            if s.startswith(tok) and len(tok) > len(found):
                found = tok
        if found != "":
            all_tokens.append(Token(found, (file, line), column, first))
            first = False
            s = s[len(found) :]
            column += len(found)
            continue

        # see if a sequence of letters and numbers
        if isnamechar(s[0]):
            i = 0
            while i < len(s) and isnamechar(s[i]):
                i += 1
            all_tokens.append(Token(s[:i], (file, line), column, first))
            first = False
            s = s[i:]
            column += i
            continue

        # string
        if s[0] == '"':
            i = 1
            str = '"'
            while i < len(s) and s[i] != '"':
                if s[i] == "\\":
                    i += 1
                    if i == len(s):
                        break
                    if s[i] == '"':
                        str += '"'
                    elif s[i] == "\\":
                        str += "\\"
                    elif s[i] == "t":
                        str += "\t"
                    elif s[i] == "n":
                        str += "\n"
                    elif s[i] == "f":
                        str += "\f"
                    elif s[i] == "r":
                        str += "\r"
                    else:
                        str += s[i]
                else:
                    str += s[i]
                i += 1
            if i < len(s):
                i += 1
            str += '"'
            all_tokens.append(Token(str, (file, line), column, first))
            first = False
            s = s[i:]
            column += i
            continue

        # everything else is a single character token
        all_tokens.append(Token(s[0], (file, line), column, first))
        first = False
        s = s[1:]
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
        if len(all_tokens) < 4:
            break
        at_least_4_before = all_tokens[0].lexeme == "----"
        tok_module = all_tokens[1].lexeme == "MODULE"
        at_least_4_after = all_tokens[3].lexeme == "----"
        if at_least_4_before and tok_module and at_least_4_after:
            break
        else:
            # Advance the line number and discard the token
            all_tokens = all_tokens[1:]

    return all_tokens
