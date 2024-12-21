import pytest

from pluspy import lexer, parser


@pytest.mark.parametrize("scannable,tok_kind", (
    ("WF_vars(proc(self))", lexer.FairnessTokenKind.WeakFairness),
    ("SF_vars(proc(self))", lexer.FairnessTokenKind.StrongFairness),
))
def test_fairness(scannable, tok_kind):
    wf_fairness = "WF_vars(proc(self))"
    sf_fairness = "SF_vars(proc(self))"
    # breakpoint()
    wf_fairness_tokens = lexer.lexer(wf_fairness, "test_parser.py")
    not_top_level = parser.GExpression(0)
    result = not_top_level.parse(wf_fairness_tokens)
