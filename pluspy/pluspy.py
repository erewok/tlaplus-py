import logging
import random
import sys

from . import ast
from .errors import PlusPyError
from .utils import FrozenDict, simplify
from .wrappers import build_wrappers

logger = logging.getLogger(__name__)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Main Class
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

class PlusPy:
    def __init__(
        self,
        filename: str,
        module_path: str,
        constants: dict | None = None,
        module_loader: ast.ModuleLoader | None = None,
        seed: int | float | str | bytes | bytearray | None = None,
        verbose: bool = False,
        silent: bool = False,
    ):
        if seed is not None:
            random.seed(seed)

        constants = constants or {}
        self.mod_loader: ast.ModuleLoader = module_loader or ast.ModuleLoader(
            module_path, modules={}, verbose=verbose, silent=silent, wrappers=build_wrappers()
        )
        # Load the module
        self.mod = ast.Module()
        if not filename.endswith(".tla"):
            filename += ".tla"

        result = self.mod.load_from_file(filename, self.mod_loader)
        if not result:
            raise PlusPyError("can't load " + filename)

        # Now that it has a name, we add it to the ModuleLoader
        self.mod_loader[self.mod.name] = self.mod

        self.constants = {self.mod.constants[k]: ast.ValueExpression(v) for (k, v) in constants.items()}

        # Substitute containers for variables
        self.containers = {v: ast.ContainerExpression(var=v) for v in self.mod.variables.values()}

    def init(self, init_op):
        op = self.mod.operators.get(init_op)
        if op is None:
            logger.error(f"Init Operator '{init_op}' not found")
            raise PlusPyError("critical failure")

        assert isinstance(op, ast.OperatorExpression), "Init Operator must be an OperatorExpression"
        assert op.args == []

        # Set the constants
        expr2 = op.expr.substitute(self.constants)

        # Replace variables with primed containers in state expressions
        ast.initializing = True
        expr3 = expr2.substitute(self.containers)
        ast.initializing = False
        r = expr3.eval(self.containers, {})
        if not r:
            logger.error(f"Initialization failed -- fatal error file={sys.stderr}")
            raise PlusPyError("critical failure")

        ok = True
        for k, v in self.containers.items():
            if v.next is None:
                logger.warning(f"UNASSIGNED: {k}")
                ok = False
        assert ok

    # This is really solving the satisfiability problem
    # However, we make only one randomized attempt
    def trynext(self, expr, args, arg):
        # set previous state to next state and next state to None
        for c in self.containers.values():
            c.prev = c.next
            c.next = None

        # Replace operator arguments with specified values
        # TODO.  Should be able to take more than 1 argument
        if len(args) > 0:
            expr = expr.substitute({args[0]: ast.ValueExpression(arg)})

        # Replace constants for their values and variables for containers
        expr2 = expr.substitute(self.constants)
        expr3 = expr2.substitute(self.containers)

        # Evaluate
        r = expr3.eval(self.containers, {})
        if r:
            error = False
            for v, c in self.containers.items():
                if c.next is None:
                    logger.error(
                        (f"Variable {v.id} did not receive a value -- " f"fatal error file={sys.stderr}"),
                    )
                    error = True
            if error:
                raise PlusPyError("Fatal error")
        else:
            for c in self.containers.values():
                c.next = c.prev
        return r

    # TODO.  Should support multiple arguments
    def next(self, next_op, arg=None):
        op = self.mod.operators.get(next_op)
        if op is None:
            logger.error(f"Next Operator '{next_op}' not found")
            raise PlusPyError("critical failure")
        assert isinstance(op, ast.OperatorExpression), "Next Operator must be an OperatorExpression"
        return self.trynext(op.expr, op.args, arg)

    # Check of state has not changed
    def unchanged(self):
        for c in self.containers.values():
            if c.next != c.prev:
                return False
        return True

    def get(self, var):
        var = self.mod.variables.get(var)
        if var is None:
            return None
        return self.containers[var].next

    def set(self, var, value):
        v = self.containers.get(self.mod.variables[var])
        v.next = value

    def getall(self):
        s = {k.id: v.next for (k, v) in self.containers.items()}
        return simplify(FrozenDict(s))
