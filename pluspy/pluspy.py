import logging
import os
import random
import sys

from . import parser
from .parser import (
    ContainerExpression,
    Module,
    ModuleLoader,
    OperatorExpression,
    ValueExpression,
)
from .utils import FrozenDict, simplify
from .wrappers import build_wrappers

logger = logging.getLogger(__name__)


def exit(status):
    sys.stdout.flush()
    os._exit(status)


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Main Class
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
class PlusPyError(Exception):
    def __init__(self, descr):
        self.descr = descr

    def __str__(self):
        return "PlusPyError: " + self.descr


class PlusPy:
    def __init__(
        self,
        filename: str,
        constants: dict | None = None,
        module_loader: ModuleLoader | None = None,
        seed=None,
        module_path: str = ".:./modules/lib:./modules/book:./modules/other",
    ):
        if seed is not None:
            random.seed(seed)

        constants = constants or {}
        self.mod_loader: ModuleLoader = module_loader or ModuleLoader({}, build_wrappers())
        self.module_path = module_path
        # Load the module
        self.mod = Module()
        if not filename.endswith(".tla"):
            filename += ".tla"
        result = self.mod.load_from_file(filename, self.mod_loader, self.module_path)
        if not result:
            raise PlusPyError("can't load " + filename)

        # Now that it has a name, we add it to the ModuleLoader
        self.mod_loader[self.mod.name] = self.mod

        self.constants = {
            self.mod.constants[k]: ValueExpression(v) for (k, v) in constants.items()
        }

        # Substitute containers for variables
        self.containers = {
            v: ContainerExpression(var=v) for v in self.mod.variables.values()
        }

    def init(self, init_op):
        op = self.mod.operators[init_op]
        assert isinstance(op, OperatorExpression)
        assert op.args == []

        # Set the constants
        expr2 = op.expr.substitute(self.constants)

        # Replace variables with primed containers in state expressions
        parser.initializing = True
        expr3 = expr2.substitute(self.containers)
        parser.initializing = False
        r = expr3.eval(self.containers, {})
        if not r:
            logger.error(f"Initialization failed -- fatal error file={sys.stderr}")
            exit(1)

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
            expr = expr.substitute({args[0]: ValueExpression(arg)})

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
                        (
                            f"Variable {v.id} did not receive a value -- "
                            f"fatal error file={sys.stderr}"
                        ),
                    )
                    error = True
            if error:
                exit(1)
        else:
            for c in self.containers.values():
                c.next = c.prev
        return r

    # TODO.  Should support multiple arguments
    def next(self, next_op, arg=None):
        op = self.mod.operators[next_op]
        assert isinstance(op, OperatorExpression)
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
