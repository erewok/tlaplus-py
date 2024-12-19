import unittest

from pluspy import lexer


class TestLexer(unittest.TestCase):
    def test_ignore_preamble(self):
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

        for case in cases:
            results = lexer.lexer(case["input"], "nofile.tla")
            a, b, _, d = results[0], results[1], results[2], results[3]
            self.assertEqual(
                "----",
                a.lexeme,
                "failed test {} expected {}, actual {}".format(case["name"], "----", a),
            )
            self.assertEqual(
                "MODULE",
                b.lexeme,
                "failed test {} expected {}, actual {}".format(case["name"], "MODULE", b),
            )
            self.assertEqual(
                "----",
                d.lexeme,
                "failed test {} expected {}, actual {}".format(case["name"], "----", a),
            )
