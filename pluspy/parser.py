import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import TypeAlias

from .lexer import InfixTokenKind, PostfixTokenKind, PrefixTokenKind, Token
from .utils import (
    isletter,
    isnamechar,
    isnumeral,
)

logger = logging.getLogger(__name__)


def exit(status):
    sys.stdout.flush()
    os._exit(status)


# When compiling and running into an identifier, it should be clear
# exactly what that identifier refers to.  It could be the name of:
#
#   - a variable
#   - a constant
#   - an operator
#   - an argument of that operator
#   - a bound variable (\E, ...)
#   - a module
#

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: various tables copied from book
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

RESERVED_WORDS = [
    "ASSUME",
    "ELSE",
    "LOCAL",
    "UNION",
    "ASSUMPTION",
    "ENABLED",
    "MODULE",
    "VARIABLE",
    "AXIOM",
    "EXCEPT",
    "OTHER",
    "VARIABLES",
    "CASE",
    "EXTENDS",
    "CHOOSE",
    "IF",
    "SUBSET",
    "WITH",
    "CONSTANT",
    "IN",
    "THEN",
    "CONSTANTS",
    "INSTANCE",
    "THEOREM",
    "DOMAIN",
    "LET",
    "UNCHANGED",
    "SF_",
    "WF_",
]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: BNF rules
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
stack = []  # Global parser state used for parsing disjuncts and conjuncts

# For error messages
shortest = []
error = []


def parse_error(a, rem: list[Token]):
    global shortest, error
    # logger.error(f"Parse error: {a} for {str(rem)}")
    if len(rem) < len(shortest):
        error = a
        shortest = rem
    return (False, a, rem)


# This isn't accurate: it's variable
ParseResult: TypeAlias = tuple[str, list[str, tuple[str, Token]], list[Token]]


def rule_match(
    name: str, found_tokens: list[Token], rule: "Rule", select: list[int] | None = None
) -> ParseResult:
    """
    Handy routine for rules that simply call other rules
    It parses s using the given rule and returns a node (name, (t, a), r)
    where name is the type of the AST node and (t, a) the result of
    parsing given the rule and r is remaining tokens.

    If t is a "Concat" rule (sequence of other rules), then you can
    select a subset using the select argument.

    Otherwise, if select is not None, the result of the rule is
    directly returned without adding a new AST node.

    :param name: the type of the AST node
    :param found_tokens: list of tokens
    :param rule: rule to parse found_tokens with
    :param select: subset of rules to select
    :return: AST node with selected rules
    """
    #   t: type of the AST node (or False if not recognized)
    #   a: contents of the AST node (or error message if t = False)
    #   r: remainder of 's' that was not parsed
    (node_type, node_content, rem) = rule.parse(found_tokens)
    if not node_type:
        return parse_error([name] + node_content, rem)
    # print(f"rule_match {name=}, {node_type=}, {node_content=}, {len(rem)=}")
    if isinstance(select, list) and node_type == "Concat":
        if len(select) == 1:
            return (name, node_content[select[0]], rem)
        return (name, [node_content[i] for i in select], rem)
    if select is not None:
        return (node_type, node_content, rem)
    return (name, (node_type, node_content), rem)


# BNF rule
class Rule(ABC):
    # found_tokens is a list of tokens.  Returns (node_type, node_content, rem) where
    #   t: type of the AST node (or False if not recognized)
    #   a: contents of the AST node (or error message if t = False)
    #   r: remainder of 's' that was not parsed
    # Must be redefined in child class
    @abstractmethod
    def parse(self, found_tokens: list[Token]) -> ParseResult:
        # return parse_error(["Rule.parse undefined"], found_tokens)
        raise NotImplementedError("Rule.parse undefined")


class GModule(Rule):
    def parse(self, found_tokens: list[Token]) -> ParseResult:
        return rule_match(
            "GModule",
            found_tokens,
            Concat(
                [
                    tok("----"),
                    tok("MODULE"),
                    Name(),
                    tok("----"),
                    Maybe(Concat([tok("EXTENDS"), CommaList(Name())]), [1]),
                    AtLeast(GUnit(), 0),
                    tok("===="),
                ]
            ),
            [2, 4, 5],
        )


# This rule recognizes a list of other rules:  rule1 & rule2 & rule3 & ...
class Concat(Rule):
    def __init__(self, what):
        self.what = what

    def parse(self, found_tokens):
        rest = found_tokens
        result = []
        for rule in self.what:
            (node_type, node_content, rem) = rule.parse(rest)
            if not node_type:
                return parse_error(["Concat"] + node_content, rem)
            result.append((node_type, node_content))
            rest = rem
        return ("Concat", result, rest)


# This rule recognizes a list of at least count rules
class AtLeast(Rule):
    def __init__(self, rule, count):
        self.rule = rule
        self.count = count

    def parse(self, found_tokens: list[Token]):
        rest = found_tokens
        result = []
        c = self.count
        while True:
            (node_type, node_content, rem) = self.rule.parse(rest)
            if not node_type:
                if c > 0:
                    return parse_error(
                        ["AtLeast" + str(self.count)] + node_content, rem
                    )
                else:
                    return ("AtLeast" + str(self.count), result, rest)
            result = result + [(node_type, node_content)]
            rest = rem
            c -= 1


# Recognizes an optional rule, i.e., 'rule?'
# 'select' can be used similarly as in Rule.rule_match()
class Maybe(Rule):
    def __init__(self, rule, select=None):
        self.rule = rule
        self.select = select

    def parse(self, found_tokens: list[Token]):
        (node_type, node_content, rem) = self.rule.parse(found_tokens)
        if not node_type:
            return ("Maybe", None, found_tokens)
        elif node_type == "Concat" and isinstance(self.select, list):
            if len(self.select) == 1:
                return ("Maybe", node_content[self.select[0]], rem)
            return ("Maybe", [node_content[i] for i in self.select], rem)
        else:
            return ("Maybe", (node_type, node_content), rem)


class tok(Rule):  # noqa: N801
    def __init__(self, what):
        self.what = what

    def parse(self, found_tokens: list[Token]):
        if found_tokens == []:
            return parse_error(["tok: no more tokens"], found_tokens)
        if found_tokens[0].lexeme == self.what:
            return ("tok", found_tokens[0], found_tokens[1:])
        return parse_error(
            [
                (
                    f"tok: no match for '{found_tokens[0].lexeme}' {found_tokens[0]}",
                    str(found_tokens[0]),
                )
            ],
            found_tokens,
        )


class Tok(Rule):
    def __init__(self, what: set[str], name: str):
        self.what = what
        self.name = name

    def parse(self, found_tokens: list[Token]):
        if found_tokens == []:
            return parse_error(["Tok: no more tokens"], found_tokens)
        if found_tokens[0].lexeme in self.what:
            return ("Tok", found_tokens[0], found_tokens[1:])
        return parse_error(["Tok: no match with " + self.name], found_tokens)


class Name(Rule):
    def __init__(self):
        pass

    def parse(self, found_tokens: list[Token]):
        if found_tokens == []:
            return parse_error(["Name"], found_tokens)
        lex = found_tokens[0].lexeme
        if lex.startswith("WF_"):
            return parse_error([("Name WF_", found_tokens[0])], found_tokens)
        if lex.startswith("SF_"):
            return parse_error([("Name SF_", found_tokens[0])], found_tokens)
        hasletter = False
        for char in lex:
            if not isnamechar(char):
                return parse_error(
                    [("Name with bad character", found_tokens[0])], found_tokens
                )
            if isletter(char):
                hasletter = True
        if hasletter:
            return ("Name", found_tokens[0], found_tokens[1:])
        return parse_error([("Name with no letter", found_tokens[0])], found_tokens)


class Identifier(Rule):
    def __init__(self):
        pass

    def parse(self, found_tokens: list[Token]):
        (node_type, node_content, rem) = Name().parse(found_tokens)
        if node_type != "Name":
            return parse_error(["Identifier: not a Name"] + node_content, found_tokens)

        lex = node_content.lexeme
        if lex in RESERVED_WORDS:
            return parse_error(
                [("Identifier: Name Reserved", node_content)], found_tokens
            )
        return ("Identifier", node_content, rem)


# Sometimes it is convenient to give certain rules names.
# A Tag node simply inserts another AST node with the given name
class Tag(Rule):
    def __init__(self, name, rule, select=None):
        self.name = name
        self.rule = rule
        self.select = select

    def parse(self, found_tokens: list[Token]):
        return rule_match(self.name, found_tokens, self.rule, self.select)


class Number(Rule):
    def __init__(self):
        pass

    def parse(self, found_tokens: list[Token]):
        if found_tokens == []:
            return parse_error(["Number"], found_tokens)
        lex = found_tokens[0].lexeme
        for c in lex:
            if not isnumeral(c):
                return parse_error([("Number", found_tokens[0])], found_tokens)
        return ("Number", lex, found_tokens[1:])


class String(Rule):
    def __init__(self):
        pass

    def parse(self, found_tokens: list[Token]):
        if found_tokens == []:
            return parse_error(["String"], found_tokens)

        lex = found_tokens[0].lexeme
        if lex[0] == '"' and lex[-1] == '"':
            return ("String", lex, found_tokens[1:])
        return parse_error([("String", found_tokens[0])], found_tokens)


class SeparatorList(Rule):
    def __init__(self, what, sep, optional):
        self.what = what  # expression to match
        self.sep = sep  # separator token
        self.optional = optional  # empty list allowed

    def parse(self, found_tokens: list[Token]):
        (node_type, node_content, rem) = self.what.parse(found_tokens)
        if not node_type:
            return (
                ("SeparatorList", [], found_tokens)
                if self.optional
                else (False, ["SeparatorList"] + node_content, rem)
            )
        rest = rem
        result = [(node_type, node_content)]
        while True:
            if rest[0].lexeme != self.sep:
                return ("SeparatorList", result, rest)
            (node_type, node_content, rem) = self.what.parse(rest[1:])
            if not node_type:
                return ("SeparatorList", result, rest)
            result.append((node_type, node_content))
            rest = rem


class CommaList(Rule):
    def __init__(self, what):
        self.what = what

    def parse(self, found_tokens: list[Token]):
        (node_type, node_content, rem) = self.what.parse(found_tokens)
        if not node_type:
            return parse_error(["CommaList"] + node_content, rem)
        rest = rem
        result = [(node_type, node_content)]
        while True:
            if rest[0].lexeme != ",":
                return ("CommaList", result, rest)
            (node_type, node_content, rem) = self.what.parse(rest[1:])
            if not node_type:
                return ("CommaList", result, rest)
            result.append((node_type, node_content))
            rest = rem


class OneOf(Rule):
    def __init__(self, what):
        self.what = what

    def parse(self, found_tokens: list[Token]):
        shortest = found_tokens  # look for shortest remainder
        result = None
        for grammar in self.what:
            (node_type, node_content, rem) = grammar.parse(found_tokens)
            if node_type is not False:
                if len(rem) < len(shortest):
                    shortest = rem
                    result = (node_type, node_content, rem)
        if result is None:
            return parse_error([("OneOf: no match", found_tokens)], found_tokens)
        return result


class Tuple(Rule):
    def __init__(self):
        pass

    def parse(self, found_tokens: list[Token]):
        return rule_match(
            "Tuple",
            found_tokens,
            Concat(
                [
                    tok("<<"),
                    # TODO.  Book does not allow empty tuples
                    Maybe(CommaList(GExpression(0))),
                    tok(">>"),
                ]
            ),
            [1],
        )


class GUnit(Rule):
    def local(self, tag, decl):
        return Tag(tag, Concat([Maybe(tok("LOCAL")), decl]), [0, 1])

    def parse(self, found_tokens: list[Token]):
        return rule_match(
            "GUnit",
            found_tokens,
            OneOf(
                [
                    GVariableDeclaration(),
                    GConstantDeclaration(),
                    self.local("decl-op", GOperatorDefinition()),
                    self.local("decl-fun", GFunctionDefinition()),
                    self.local("decl-inst", GInstance()),
                    self.local("decl-mod", GModuleDefinition()),
                    GAssumption(),
                    GTheorem(),
                    GModule(),
                    GDivider(),
                ]
            ),
            True,
        )


class GDivider(Rule):
    def parse(self, found_tokens: list[Token]):
        return rule_match("GDivider", found_tokens, tok("----"))


class GVariableDeclaration(Rule):
    def parse(self, found_tokens: list[Token]):
        return rule_match(
            "GVariableDeclaration",
            found_tokens,
            Concat(
                [OneOf([tok("VARIABLE"), tok("VARIABLES")]), CommaList(Identifier())]
            ),
            [1],
        )


class GConstantDeclaration(Rule):
    def parse(self, found_tokens: list[Token]):
        return rule_match(
            "GConstantDeclaration",
            found_tokens,
            Concat([OneOf([tok("CONSTANT"), tok("CONSTANTS")]), CommaList(GOpDecl())]),
            [1],
        )


class GOpDecl(Rule):
    def parse(self, found_tokens: list[Token]):
        return rule_match(
            "GOpDecl",
            found_tokens,
            OneOf(
                [
                    Identifier(),
                    Tag(
                        "paramOp",
                        Concat([Identifier(), tok("("), CommaList(tok("_")), tok(")")]),
                        [0, 2],
                    ),
                    Tag(
                        "prefixOp",
                        Concat(
                            [
                                Tok(PrefixTokenKind.token_set(), "prefix operator"),
                                tok("_"),
                            ]
                        ),
                        [0],
                    ),
                    Tag(
                        "infixOp",
                        Concat(
                            [
                                tok("_"),
                                Tok(InfixTokenKind.token_set(), "infix operator"),
                                tok("_"),
                            ]
                        ),
                        [1],
                    ),
                    Tag(
                        "postfixOp",
                        Concat(
                            [
                                tok("_"),
                                Tok(PostfixTokenKind.token_set(), "postfix operator"),
                            ]
                        ),
                        [1],
                    ),
                ]
            ),
        )


class GNonFixLHS(Rule):
    def parse(self, s):
        return rule_match(
            "GNonFixLHS",
            s,
            Concat(
                [
                    Identifier(),
                    Maybe(Concat([tok("("), CommaList(GOpDecl()), tok(")")]), [1]),
                ]
            ),
            [0, 1],
        )


class GFunctionDefinition(Rule):
    def parse(self, s):
        return rule_match(
            "GFunctionDefinition",
            s,
            Concat(
                [
                    Identifier(),
                    tok("["),
                    CommaList(GQuantifierBound()),
                    tok("]"),
                    tok("=="),
                    GExpression(0),
                ]
            ),
            [0, 2, 5],
        )


class GOperatorDefinition(Rule):
    def parse(self, s):
        return rule_match(
            "GOperatorDefinition",
            s,
            Concat(
                [
                    OneOf(
                        [
                            GNonFixLHS(),
                            Tag(
                                "prefix",
                                Concat(
                                    [
                                        Tok(
                                            PrefixTokenKind.token_set(),
                                            "prefix operator",
                                        ),
                                        Identifier(),
                                    ]
                                ),
                            ),
                            Tag(
                                "infix",
                                Concat(
                                    [
                                        Identifier(),
                                        Tok(
                                            InfixTokenKind.token_set(), "infix operator"
                                        ),
                                        Identifier(),
                                    ]
                                ),
                            ),
                            Tag(
                                "postfix",
                                Concat(
                                    [
                                        Identifier(),
                                        Tok(
                                            PostfixTokenKind.token_set(),
                                            "postfix operator",
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                    tok("=="),
                    GExpression(0),
                ]
            ),
            [0, 2],
        )


class GTheorem(Rule):
    def parse(self, s):
        return rule_match("GTheorem", s, Concat([tok("THEOREM"), GExpression(0)]), [1])


class GAssumption(Rule):
    def parse(self, s):
        return rule_match(
            "GAssumption",
            s,
            Concat(
                [
                    OneOf([tok("ASSUME"), tok("ASSUMPTION"), tok("AXIOM")]),
                    Maybe(Concat([Identifier(), tok("==")])),
                    GExpression(0),
                ]
            ),
            [1],
        )


class IdentifierOrTuple(Rule):
    def parse(self, s):
        return rule_match(
            "IdentifierOrTuple",
            s,
            OneOf(
                [
                    Identifier(),
                    Tag(
                        "Tuple",
                        Concat([tok("<<"), CommaList(Identifier()), tok(">>")]),
                        [1],
                    ),
                ]
            ),
            [0],
        )


class GQuantifierBound(Rule):
    def parse(self, s):
        return rule_match(
            "GQuantifierBound",
            s,
            Concat(
                [OneOf([CommaList(Identifier()), Tuple()]), tok("\\in"), GExpression(0)]
            ),
            [0, 2],
        )


class GInstance(Rule):
    def parse(self, s):
        return rule_match(
            "GInstance",
            s,
            Concat(
                [
                    tok("INSTANCE"),
                    Name(),
                    Maybe(Concat([tok("WITH"), CommaList(GSubstitution())]), [1]),
                ]
            ),
            [1, 2],
        )


class GSubstitution(Rule):
    def parse(self, s):
        return rule_match(
            "GSubstitution",
            s,
            Concat(
                [
                    # TODO.  Can also replace prefix, infix, or postfix ops
                    Identifier(),
                    tok("<-"),
                    GArgument(),
                ]
            ),
            [0, 2],
        )


class GArgument(Rule):
    def parse(self, s):
        return rule_match(
            "GArgument",
            s,
            OneOf(
                [
                    GExpression(0),
                    Tag("arg-prefix", GGeneralPrefixOp()),
                    Tag("arg-infix", GGeneralInfixOp()),
                    Tag("arg-postfix", GGeneralPostfixOp()),
                ]
            ),
        )


class GInstancePrefix(Rule):
    def parse(self, s):
        return rule_match(
            "GInstancePrefix",
            s,
            AtLeast(
                Concat(
                    [
                        Identifier(),
                        Maybe(
                            Concat(
                                [
                                    tok("("),
                                    # TODO. book has GExpression here, but seems wrong
                                    CommaList(GArgument()),
                                    tok(")"),
                                ]
                            ),
                            [1],
                        ),
                        tok("!"),
                    ]
                ),
                0,
            ),
        )


class GGeneralIdentifier(Rule):
    def parse(self, s):
        return rule_match(
            "GGeneralIdentifier", s, Concat([GInstancePrefix(), Identifier()])
        )


class GGeneralPrefixOp(Rule):
    def parse(self, s):
        return rule_match(
            "GGeneralPrefixOp",
            s,
            Concat(
                [GInstancePrefix(), Tok(PrefixTokenKind.token_set(), "prefix operator")]
            ),
        )


class GGeneralInfixOp(Rule):
    def parse(self, s):
        return rule_match(
            "GGeneralInfixOp",
            s,
            Concat(
                [GInstancePrefix(), Tok(InfixTokenKind.token_set(), "infix operator")]
            ),
        )


class GGeneralPostfixOp(Rule):
    def parse(self, s):
        return rule_match(
            "GGeneralPostfixOp",
            s,
            Concat(
                [
                    GInstancePrefix(),
                    Tok(PostfixTokenKind.token_set(), "postfix operator"),
                ]
            ),
        )


class GModuleDefinition(Rule):
    def parse(self, s):
        return rule_match(
            "GModuleDefinition",
            s,
            Concat([GNonFixLHS(), tok("=="), GInstance()]),
            [0, 2],
        )


class GExpression(Rule):
    def __init__(self, level):
        self.level = level

    def parse(self, found_tokens: list[Token]):
        if found_tokens == []:
            return parse_error(["GExpression: empty list"], found_tokens)

        # If at the top precedence level, get a basic expression.
        if self.level == 18:
            return rule_match("GExpression18", found_tokens, GBasicExpression(), True)

        # See if this is an expression starting with /\ or \/
        first = found_tokens[0]
        lex = first.lexeme
        if lex in {"/\\", "\\/"}:
            lex: str = first.lexeme
            column: int = first.column
            token: tuple[str, int, bool] = (lex, column, True)
            stack.append(token)
            (node_type, node_content, rem) = GExpression(0).parse(found_tokens[1:])
            if node_type is False:
                stack.pop()
                return parse_error(
                    [(f"GExpression{self.level}", first), *node_content], rem
                )

            while rem != [] and rem[0].junct == token:
                (node_type2, node_content2, rem2) = GExpression(0).parse(rem[1:])
                if not node_type2:
                    stack.pop()
                    return parse_error(["GExpression0", *node_content2], rem2)
                (node_type, node_content, rem) = (
                    "Infix0",
                    (first, (node_type, node_content), (node_type2, node_content2)),
                    rem2,
                )
            stack.pop()
            return (node_type, node_content, rem)

        # See if the expression starts with a prefix operator.
        # TODO.  Should match again GGeneralPrefixOp
        token_kind = PrefixTokenKind.get(found_tokens[0].lexeme)
        if token_kind is not None:
            # Compute the precedence level of the operator.
            prec = token_kind.precedence()

            # Parse an expression of the given precedence level.
            (node_type, node_content, rem) = GExpression(prec).parse(found_tokens[1:])
            if node_type is False:
                return parse_error(
                    ["GExpression" + str(self.level) + ": " + str(found_tokens[0])] + node_content,
                    rem,
                )
            (node_type, node_content, rem) = (
                "Prefix" + str(prec),
                (found_tokens[0], (node_type, node_content)),
                rem,
            )

        # If not a prefix get an expression at the next precedence level
        else:
            (node_type, node_content, rem) = GExpression(self.level + 1).parse(found_tokens)
            if node_type is False:
                return parse_error(
                    ["GExpression" + str(self.level) + ": " + str(found_tokens[0])] + node_content,
                    rem,
                )

        # Loop through the remainder.
        while rem != []:
            # If a disjunct or conjunct, we're done.
            if rem[0].junct in stack:
                return (node_type, node_content, rem)

            # See if it's a postfix expression with sufficient precedence
            token_kind = PostfixTokenKind.get(rem[0].lexeme)
            if token_kind is not None:
                # Compute the precedence level.  If of a lower level, we're done.
                prec = token_kind.precedence()
                if prec <= self.level:
                    return (node_type, node_content, rem)

                # Check for an index expression
                if rem[0].lexeme == "[":
                    (node_type2, node_content2, rem2) = Concat(
                        [tok("["), CommaList(GExpression(0)), tok("]")]
                    ).parse(rem)
                    if not node_type2:
                        return (
                            False,
                            ["GExpresssion" + str(self.level) + ": bad index"]
                            + node_content2,
                            rem2,
                        )
                    (node_type, node_content, rem) = (
                        "Index",
                        (rem[0], (node_type, node_content), node_content2[1]),
                        rem2,
                    )
                else:
                    (node_type, node_content, rem) = (
                        "Postfix" + str(self.level),
                        ((node_type, node_content), rem[0]),
                        rem[1:],
                    )

            else:
                # See if the next token is an infix operator.  If not, we're done.
                lex = rem[0].lexeme
                token_kind = InfixTokenKind.get(lex)
                if token_kind is None:
                    return (node_type, node_content, rem)

                # If it's the '.' operator, it should be followed by a field name
                if lex == ".":
                    (node_type2, node_content2, rem2) = Name().parse(rem[1:])
                    if node_type2 is False:
                        return (
                            False,
                            ["GExpression" + str(self.level) + ": no field name"]
                            + node_content2,
                            rem2,
                        )
                    assert node_type2 == "Name"
                    (node_type, node_content, rem) = (
                        "Index",
                        (
                            rem[0],
                            (node_type, node_content),
                            ("CommaList", [("String", f'"{node_content2.lexeme}"')]),
                        ),
                        rem2,
                    )

                else:
                    # Compute the precedence.  If too low, we're done.
                    prec = token_kind.precedence()
                    if prec <= self.level:
                        return (node_type, node_content, rem)

                    # Get the next expression at that precedence level.
                    (node_type2, node_content2, rem2) = GExpression(prec).parse(rem[1:])
                    if node_type2 is False:
                        return (
                            False,
                            ["GExpression" + str(self.level) + ": " + str(rem[0])]
                            + node_content2,
                            rem2,
                        )

                    # Cartesian products are not infix operators
                    if lex in {"\\X", "\\times"}:
                        if node_type == "Cartesian":
                            (node_type, node_content, rem) = (
                                "Cartesian",
                                node_content + [(node_type2, node_content2)],
                                rem2,
                            )
                        else:
                            (node_type, node_content, rem) = (
                                "Cartesian",
                                [
                                    (node_type, node_content),
                                    (node_type2, node_content2),
                                ],
                                rem2,
                            )
                    else:
                        (node_type, node_content, rem) = (
                            "Infix" + str(self.level),
                            (
                                rem[0],
                                (node_type, node_content),
                                (node_type2, node_content2),
                            ),
                            rem2,
                        )
        return (node_type, node_content, rem)


# Separate AST node for the EXCEPT clause in a function update operation
class GExcept(Rule):
    def parse(self, s):
        (node_type, node_content, rem) = CommaList(
            Concat(
                [
                    tok("!"),
                    AtLeast(
                        OneOf(
                            [
                                Tag("efield", Concat([tok("."), Name()]), [1]),
                                Tag(
                                    "elist",
                                    Concat(
                                        [
                                            tok("["),
                                            CommaList(GExpression(0)),
                                            tok("]"),
                                        ]
                                    ),
                                    [1],
                                ),
                            ]
                        ),
                        1,
                    ),
                    tok("="),
                    GExpression(0),
                ]
            )
        ).parse(s)
        if not node_type:
            return (False, ["GExcept"] + node_content, rem)
        assert node_type == "CommaList"
        result = []
        for x in node_content:
            (t2, a2) = x
            assert t2 == "Concat"
            (t3, a3) = a2[1]
            assert t3 == "AtLeast1"
            result = result + [(a3, a2[3])]
        return ("GExcept", result, rem)


class GBasicExpression(Rule):
    def parse(self, s):
        return rule_match(
            "GBasicExpression",
            s,
            OneOf(
                [
                    Tag(
                        "op",
                        Concat(
                            [
                                GGeneralIdentifier(),
                                Maybe(
                                    Concat(
                                        [tok("("), CommaList(GArgument()), tok(")")]
                                    ),
                                    [1],
                                ),
                            ]
                        ),
                        [0, 1],
                    ),
                    Tag(
                        "parentheses", Concat([tok("("), GExpression(0), tok(")")]), [1]
                    ),
                    Tag(
                        "exists",
                        Concat(
                            [
                                tok("\\E"),
                                CommaList(GQuantifierBound()),
                                tok(":"),
                                GExpression(0),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "forall",
                        Concat(
                            [
                                tok("\\A"),
                                CommaList(GQuantifierBound()),
                                tok(":"),
                                GExpression(0),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "temporal_exists",
                        Concat(
                            [
                                tok("\\EE"),
                                CommaList(Identifier()),
                                tok(":"),
                                GExpression(0),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "temporal_forall",
                        Concat(
                            [
                                tok("\\AA"),
                                CommaList(Identifier()),
                                tok(":"),
                                GExpression(0),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tuple(),
                    Tag(
                        "set",
                        Concat(
                            [tok("{"), Maybe(CommaList(GExpression(0))), tok("}")]
                        ),
                        [1],
                    ),
                    Tag(
                        "filter",
                        Concat(
                            [
                                tok("{"),
                                IdentifierOrTuple(),
                                tok("\\in"),
                                GExpression(0),
                                tok(":"),
                                GExpression(0),
                                tok("}"),
                            ]
                        ),
                        [1, 3, 5],
                    ),
                    Tag(
                        "gen",
                        Concat(
                            [
                                tok("{"),
                                GExpression(0),
                                tok(":"),
                                CommaList(GQuantifierBound()),
                                tok("}"),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "square",
                        Concat([tok("["), GExpression(0), tok("]_"), GExpression(0)]),
                        [1, 3],
                    ),
                    Tag(
                        "lambda",
                        Concat(
                            [
                                tok("["),
                                CommaList(GQuantifierBound()),
                                tok("|->"),
                                GExpression(0),
                                tok("]"),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "except",
                        Concat(
                            [
                                tok("["),
                                GExpression(0),
                                tok("EXCEPT"),
                                GExcept(),
                                tok("]"),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "funcset",
                        Concat(
                            [
                                tok("["),
                                GExpression(0),
                                tok("->"),
                                GExpression(0),
                                tok("]"),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "recorddef",
                        Concat(
                            [
                                tok("["),
                                CommaList(Concat([Name(), tok(":"), GExpression(0)])),
                                tok("]"),
                            ]
                        ),
                        [1],
                    ),
                    Tag(
                        "recordvalue",
                        Concat(
                            [
                                tok("["),
                                CommaList(Concat([Name(), tok("|->"), GExpression(0)])),
                                tok("]"),
                            ]
                        ),
                        [1],
                    ),
                    Tag(
                        "choose",
                        Concat(
                            [
                                tok("CHOOSE"),
                                Identifier(),
                                Maybe(Concat([tok("\\in"), GExpression(0)]), [1]),
                                tok(":"),
                                GExpression(0),
                            ]
                        ),
                        [1, 2, 4],
                    ),
                    Tag(
                        "if",
                        Concat(
                            [
                                tok("IF"),
                                GExpression(0),
                                tok("THEN"),
                                GExpression(0),
                                tok("ELSE"),
                                GExpression(0),
                            ]
                        ),
                        [1, 3, 5],
                    ),
                    Tag(
                        "case",
                        Concat(
                            [
                                tok("CASE"),
                                SeparatorList(
                                    Concat([GExpression(0), tok("->"), GExpression(0)]),
                                    "[]",
                                    False,
                                ),
                                Maybe(
                                    Concat(
                                        [
                                            tok("[]"),
                                            tok("OTHER"),
                                            tok("->"),
                                            GExpression(0),
                                        ]
                                    ),
                                    [3],
                                ),
                            ]
                        ),
                        [1, 2],
                    ),
                    Tag(
                        "let",
                        Concat(
                            [
                                tok("LET"),
                                AtLeast(
                                    OneOf(
                                        [
                                            GOperatorDefinition(),
                                            GFunctionDefinition(),
                                            GModuleDefinition(),
                                        ]
                                    ),
                                    1,
                                ),
                                tok("IN"),
                                GExpression(0),
                            ]
                        ),
                        [1, 3],
                    ),
                    # There's an ambiguity for WF_a(b): does it mean
                    # "WF_ a(b)" or "WF_a (b)"?  My parser gets confused
                    # so I restricted it a bit
                    Tag(
                        "wf",
                        Concat(
                            [
                                tok("WF_"),
                                IdentifierOrTuple(),
                                tok("("),
                                GExpression(0),
                                tok(")"),
                            ]
                        ),
                        [1, 3],
                    ),
                    Tag(
                        "sf",
                        Concat(
                            [
                                tok("SF_"),
                                IdentifierOrTuple(),
                                tok("("),
                                GExpression(0),
                                tok(")"),
                            ]
                        ),
                        [1, 3],
                    ),
                    Number(),
                    String(),
                    Tag("at", tok("@")),
                ]
            ),
        )
