import pytest

from pluspy import lexer

cases = [
    {
        "name": "preamble",
        "input": r"""
                    This is some prose preceding the module definition.

                    \* WORKAROUND: Comment prose before the module definition.

                    ---- MODULE AsyncGameOfLifeDistributed -----

                    VARIABLE x
                    Spec == x = TRUE /\ [][x'\in BOOLEAN]_x
                    ====
                    """,
    },
    {
        "name": "preamble with four dashes",
        "input": r"""
                    ---- What is this

                    \* A comment

                    And more preamble there is.

                    ------------------------------- MODULE Somename -------------------------------
                    """,
    },
    {
        "name": "preamble with commented module",
        "input": r"""'
                    ---- What is this

                    \* A comment
                    \* ---- MODULE foo ----
                    And more preamble there is.

                    -------------------------------  MODULE Foo -------------------------------


                    ================================ =============================================
        """,
    },
]


@pytest.mark.parametrize("case", cases)
def test_ignore_preamble(case):
    results = lexer.lexer(case["input"], "nofile.tla")
    a, b, _, d = results[0], results[1], results[2], results[3]
    assert a.lexeme == "----"
    assert b.lexeme == "MODULE"
    assert d.lexeme == "----"
    assert d.line == 8
    assert d.column == 20


@pytest.mark.parametrize("scannable,tok_kind", (
    ("WF_vars(proc(self))", lexer.FairnessTokenKind.WeakFairness),
    ("SF_vars(proc(self))", lexer.FairnessTokenKind.StrongFairness),
))
def test_fairness(scannable, tok_kind):
    result = lexer.FairnessTokenKind.scan_fairness(scannable)
    expected = (tok_kind, 'vars')
    assert result == expected

    tokens = lexer.lexer(scannable, "text_lexer.py")
    assert len(tokens) == 7
    assert isinstance(tokens[0].kind, lexer.FairnessTokenKind)
    assert tokens[1].kind == lexer.TokenKind.ParensStart
    assert tokens[-1].kind == lexer.TokenKind.ParensEnd
    assert tokens[-2].kind == lexer.TokenKind.ParensEnd
