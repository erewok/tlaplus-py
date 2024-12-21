import logging
import os
import random
import sys
import traceback
from typing import Optional

from . import run_global_vars, wrappers
from .errors import PlusPyError
from .lexer import lexer_discard_preamble, Token
from .parser import GModule
from .utils import (
    convert,
    FrozenDict,
    key,
    Nonce,
    val_to_string,
)

logger = logging.getLogger(__name__)
# In order to do this mapping, we keep a stack of dictionaries
# that map names to expressions for these things.
name_stack = [{}]

# kludge: as object definitions are properly nested, I maintain a stack
# of modules
modstk = []

# For debugging, we give each bounded variable a unique identifier
bv_counter = 0

# kludge: for transforming expression for initialization
initializing = False


def tokens_to_string(tokens):
    return ", ".join(map(lambda t: f"\n{str(t)}" if t.first else str(t), tokens))


# Find a file using a directory path
def file_find(name, path):
    sep = ";" if path.find(";") >= 0 else ":"
    for dir in path.split(sep):
        full = os.path.join(dir, name)
        if os.path.exists(full):
            return os.path.abspath(full)
    return False


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Module specification
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
class ModuleLoader:
    __slots__ = ["loaded_modules", "wrappers", "module_path", "verbose", "silent"]

    def __init__(
        self,
        module_path: str,
        modules: dict | None = None,
        verbose: bool = False,
        silent: bool = False,
        wrappers: dict[str, dict[str, "wrappers.Wrapper"]] | None = None,
    ):
        self.module_path = module_path
        self.verbose = verbose
        self.silent = silent
        self.loaded_modules: dict[str, Module] = modules
        self.wrappers = wrappers

    def get(self, key: str, default=None) -> Optional["Module"]:
        return self.loaded_modules.get(key, default)

    def __getitem__(self, key: str) -> "Module":
        return self.loaded_modules[key]

    def __setitem__(self, key: str, value: "Module") -> None:
        self.loaded_modules[key] = value
        return None

    def get_mod_wrapper(self, module: str) -> dict[str, "wrappers.Wrapper"] | None:
        return self.wrappers.get(module)


class Module:
    def __init__(self):
        self.name = None
        self.constants = dict()  # name -> ConstantExpression
        self.variables = dict()  # name -> VariableExpression
        self.operators = dict()  # name -> OperatorExpression
        self.wrappers = dict()  # name -> BuiltinExpression
        self.globals = set()  # set of non-LOCAL names

    def __str__(self):
        op_names = ", ".join(self.operators.keys())
        return f"Module({self.name}, constants={self.constants}, vars={self.variables}, operators={op_names})"

    # handle a CONSTANT declaration
    def compile_constant_declaration(self, ast):
        (t0, a0) = ast
        assert t0 == "CommaList"
        for t1, a1 in a0:
            assert t1 == "GOpDecl"
            (t2, a2) = a1
            if t2 == "Identifier":
                id = a2.lexeme
                nargs = 0
            elif t2 == "paramOp":
                (t3, a3) = a2[0]
                assert t3 == "Identifier"
                (t4, a4) = a2[1]
                assert t4 == "CommaList"
                id = a3.lexeme
                nargs = len(a4)
            elif t2 == "prefixOp" or t2 == "postfixOp":
                id = a2
                nargs = 1
            elif t2 == "infixOp":
                id = a2
                nargs = 2
            else:
                raise AssertionError("Invalid constant")
            ce = ConstantExpression(id, nargs)
            self.constants[id] = ce
            name_stack[-1][id] = ce

    # handle a VARIABLE declaration
    def compile_variable_declaration(self, tree):
        (t, a) = tree
        assert t == "CommaList"
        for t2, a2 in a:
            assert t2 == "Identifier"
            id = a2.lexeme
            ve = VariableExpression(id)
            self.variables[id] = ve
            name_stack[-1][id] = ve

    # handle an "Operator == INSTANCE name" definition
    def compile_module_definition(self, md, is_global: bool, mod_loader: ModuleLoader):
        (t0, a0) = md[0]
        assert t0 == "GNonFixLHS"
        assert len(a0) == 2
        inst = md[1]

        (t2, a2) = a0[0]
        assert t2 == "Identifier"
        id = a2.lexeme
        (t3, a3) = a0[1]
        assert t3 == "Maybe"
        if a3 is None:
            args = []
        else:
            (t4, a4) = a3
            assert t4 == "CommaList"
            args = a4

        cargs = []
        for t, a in args:
            assert t == "GOpDecl"
            (t2, a2) = a
            if t2 == "Identifier":
                cargs = cargs + [(a2.lexeme, 0)]
            elif t2 == "paramOp":
                (t3, a3) = a2[0]
                assert t3 == "Identifier"
                (t4, a4) = a2[1]
                assert t4 == "CommaList"
                cargs = cargs + [(a3.lexeme, len(a4))]
            elif t2 == "prefixOp" or t2 == "postfixOp":
                cargs = cargs + [(a2, 1)]
            elif t2 == "infixOp":
                cargs = cargs + [(a2, 2)]
            else:
                raise AssertionError("Invalid argument")

        mi = ModInst()
        args = [ArgumentExpression(a, c) for (a, c) in cargs]
        name_stack.append({a.id: a for a in args})
        mi.compile(inst, mod_loader)
        name_stack.pop()

        # We put the ModInst inside the expr field of an OperatorExpression
        od = OperatorExpression(id=id, args=args, expr=mi)
        self.operators[id] = od
        if is_global:
            self.globals.add(id)
        name_stack[-1][id] = od
        if mod_loader.verbose:
            logger.info(f"++> {od}, {mi}")

    # handle the next TLA "Unit" in the source
    def compile_unit(self, tree, mod_loader: ModuleLoader):
        (t, a) = tree
        if t == "GVariableDeclaration":
            self.compile_variable_declaration(a)
        elif t == "GConstantDeclaration":
            self.compile_constant_declaration(a)
        elif t == "decl-op":
            (tloc, aloc) = a[0]
            assert tloc == "Maybe"
            (t1, a1) = a[1]
            assert t1 == "GOperatorDefinition"
            (ident, args, expr) = compile_operator_definition(a1)
            if ident in self.wrappers.keys():
                od = OperatorExpression(
                    ident,
                    args,
                    BuiltinExpression(ident, args, self.wrappers[ident]),
                )
            else:
                od = OperatorExpression(ident, args, expr)
            self.operators[ident] = od
            if aloc is None:
                self.globals.add(ident)
            name_stack[-1][ident] = od.expr if args == [] else od
            if mod_loader.verbose:
                logger.info(f"+-> {ident=}, {args=}, {expr.primed=}\n\t{expr=}")
        elif t == "decl-inst":
            (tloc, aloc) = a[0]
            assert tloc == "Maybe"
            mi = ModInst()
            mi.compile(a[1], mod_loader)
            for k in mi.globals:
                self.operators[k] = mi.operators[k]
                if aloc is None:
                    self.globals.add(k)
        elif t == "decl-fun":
            (tloc, aloc) = a[0]
            assert tloc == "Maybe"
            (t1, a1) = a[1]
            assert t1 == "GFunctionDefinition"
            (id, args, expr) = compile_function_definition(a1)
            od = OperatorExpression(id, args, expr)
            self.operators[id] = od
            if aloc is None:
                self.globals.add(id)
            assert args == []
            # name_stack[-1][id] = od
            name_stack[-1][id] = expr
            if mod_loader.verbose:
                logger.info(f"++> {id=}, {args=}, {expr.primed=}\n\t{expr=}")
        elif t == "decl-mod":
            (tloc, aloc) = a[0]
            assert tloc == "Maybe"
            (t1, a1) = a[1]
            assert t1 == "GModuleDefinition"
            self.compile_module_definition(a1, tloc is not None, mod_loader)
        elif t in {"GTheorem", "GAssumption", "GDivider"}:
            pass
        elif t == "GModule":
            mod = Module()
            mod.compile(tree, mod_loader)
            name_stack[-1][mod.name] = mod
        else:
            logger.error(
                f"Fail compile_unit {tree=}",
            )
            raise AssertionError("Invalid unit")

    # Get operators from EXTENDS clause
    def extends(self, ast, mod_loader: ModuleLoader):
        for n, m in ast:
            assert n == "Name"
            mod = load_module(m.lexeme, mod_loader)
            assert mod.constants == dict()
            assert mod.variables == dict()
            for k in mod.globals:
                self.operators[k] = mod.operators[k]
                self.globals.add(k)
                if mod.wrappers.get(k) is not None:
                    self.wrappers[k] = mod.wrappers[k]
                name_stack[-1][k] = mod.operators[k]

    # Given AST, handle all the TLA+ units in the AST
    def compile(self, tree, mod_loader: ModuleLoader):
        (t, a) = tree
        if t is False:
            return False
        assert t == "GModule"
        assert len(a) == 3
        (t0, a0) = a[0]
        assert t0 == "Name"
        self.name = a0.lexeme

        # Set wrappers
        self.wrappers = mod_loader.get_mod_wrapper(self.name)
        if self.wrappers is None:
            self.wrappers = {}

        # Add a new dictionary to the name stack
        name_stack.append({})

        (t1, a1) = a[1]
        assert t1 == "Maybe"
        if a1 is not None:
            (tx, ax) = a1
            assert tx == "CommaList"
            self.extends(ax, mod_loader)

        (t2, a2) = a[2]
        assert t2 == "AtLeast0"
        for ast2 in a2:
            self.compile_unit(ast2, mod_loader)

        if mod_loader.verbose:
            logger.info(f"{self.name} Variables: {self.variables}")

        name_stack.pop()
        return True

    # Load and compile the given TLA+ source, which is a string
    def load_from_string(self, source_str, srcid, mod_loader: ModuleLoader):
        # First run source through lexical analysis
        tokens: list[Token] = lexer_discard_preamble(source_str, srcid)

        if mod_loader.verbose:
            logger.info("---------------")
            logger.info("Output from Lexer")
            logger.info("---------------")
            logger.info(tokens_to_string(tokens))

        # Parse the output from the lexer into an AST
        gmod = GModule()

        # Error handling
        global shortest, error
        shortest = tokens

        (node_type, node_content, rem) = gmod.parse(tokens)
        # [0] type of the AST root node (False if error)
        # [1] the content (or error message list if error)
        # [2] is the suffix of the lexer output that could not be parsed

        if node_type is False:
            logger.error(f"Parsing failed {node_content}")
            logger.error(print_ast((node_type, node_content), 2))
            logger.error(f"At position {shortest[0]}")
            return False

        if rem != []:
            logger.info(f"Remainder {rem[0]}")

        # Handle all TLA+ units in the AST
        if mod_loader.verbose:
            logger.info("---------------")
            splitted = source_str.split("\n")[0].replace("-", "")
            logger.info(f"Compile {splitted}")
            logger.info("---------------")

        modstk.append(self)
        result = self.compile((node_type, node_content), mod_loader)
        modstk.pop()

        return result

    def load(self, f, srcid, mod_loader: ModuleLoader):
        all = ""
        for line in f:
            all += line
        return self.load_from_string(all, srcid, mod_loader)

    def load_from_file(self, file, mod_loader: ModuleLoader):
        full = file_find(file, mod_loader.module_path)
        if not full:
            return False
        with open(full) as f:
            return self.load(f, file, mod_loader)


def load_module(name: str, mod_loader: ModuleLoader):
    mod = name_lookup(name)
    if mod is False:
        if mod := mod_loader.get(name):
            return mod

    mod = Module()
    name_stack.append({})
    if not mod.load_from_file(name + ".tla", mod_loader):
        logger.error(f"can't load {name}: fatal error file={sys.stderr}")
        raise PlusPyError("critical failure")
    name_stack.pop()
    mod_loader[name] = mod
    return mod


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Module instance
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


# Describes an "INSTANCE module-name WITH k <- e ..." expression.
# Here each k is either a constant or variable name of the module, and e
# some expression that should be substituted for it
class ModInst:
    def __init__(
        self,
        module=None,
        substitutions=None,
        operators=None,
        wrappers=None,
        globals=None,
    ):
        self.module = module
        self.substitutions = substitutions
        self.operators = operators
        self.wrappers = wrappers
        self.globals = globals
        self.primed = False

    def __str__(self):
        subs = ""
        for k, v in self.substitutions.items():
            if subs != "":
                subs += ", "
            subs += str(k) + ": " + str(v)
        return "Instance(" + self.module.name + ", [" + subs + "])"

    def __repr__(self):
        return self.__str__()

    def substitute(self, subs):
        substitutions = {k: v.substitute(subs) for (k, v) in self.substitutions.items()}
        return ModInst(
            module=self.module,
            substitutions=substitutions,
            operators={},
            globals=set(),
        )

    def set(self, module, substitutions):
        self.module = module
        self.substitutions = substitutions

    def compile(self, tree, mod_loader: ModuleLoader):
        (t, a) = tree
        assert t == "GInstance"
        (t1, a1) = a[0]
        assert t1 == "Name"
        self.module = load_module(a1.lexeme, mod_loader)

        (t2, a2) = a[1]
        assert t2 == "Maybe"
        d = {}
        if a2 is not None:
            (t3, a3) = a2
            assert t3 == "CommaList"
            for t4, a4 in a3:
                assert t4 == "GSubstitution"
                (t5, a5) = a4[0]
                assert t5 == "Identifier"
                (t6, a6) = a4[1]
                assert t6 == "GArgument"
                d[a5.lexeme] = compile_expression(a6)

        # We now need to replace all the constants and variables in the
        # operators of the module.  Some may have been specified using
        # WITH (captured in 'd'), others are implicit.

        self.substitutions = {}
        for k, v in self.module.constants.items():
            if k in d.keys():
                self.substitutions[v] = d[k]
            else:
                self.substitutions[v] = name_find(k)
        for k, v in self.module.variables.items():
            if k in d.keys():
                self.substitutions[v] = d[k]
            else:
                self.substitutions[v] = name_find(k)
        self.operators = {}
        self.globals = set()
        self.wrappers = {}
        for k in self.module.globals:
            assert k not in self.globals
            assert self.operators.get(k) is None
            d = self.module.operators[k]
            self.operators[k] = d.substitute(self.substitutions)
            self.globals.add(k)
            if self.module.wrappers.get(k) is not None:
                self.wrappers[k] = self.module.wrappers[k]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: Expressions
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


def name_lookup(name):
    for d in reversed(name_stack):
        if ex := d.get(name):
            return ex
    return False


# Like name_lookup but prints an error
def name_find(name):
    e = name_lookup(name)
    if not e:
        logger.error(f"Identifier {name} not found")
    return e


# Get the prefix of an A!B!C type expression
def getprefix(ast, operators):
    (t, a) = ast
    assert t == "GInstancePrefix"
    (t1, a1) = a
    assert t1 == "AtLeast0"
    instances = []
    for x in a1:
        (t2, a2) = x
        assert t2 == "Concat"
        assert len(a2) == 3
        (t3, a3) = a2[0]
        assert t3 == "Identifier"
        od = operators[a3.lexeme]
        assert isinstance(od, OperatorExpression)
        if not isinstance(od.expr, ModInst):
            print("trying to instantiate", od.expr)
        assert isinstance(od.expr, ModInst)
        (t4, a4) = a2[1]
        assert t4 == "Maybe"
        args = []
        if a4 is not None:
            (t5, a5) = a4
            assert t5 == "CommaList"
            for t, a in a5:
                assert t == "GArgument"
                args += [compile_expression(a)]
        instances += [(a3, od, args)]
        operators = od.expr.operators
    return (operators, instances)


# handle complicated situations like A(a)!B(b)!C(c)
# This is best done backwards:
#   First find A!B!C
#   Make substitutions to create A!B!C(c)
#   Then A!B(b)!C(c)
#   Finally A(a)!B(b)!C(c)
def op_subst(instances):
    (lex, iop, iargs) = instances[0]
    assert isinstance(iop, OperatorExpression)
    oargs = iop.args
    oexpr = iop.expr

    if len(instances) == 1:
        expr = oexpr
    else:
        assert isinstance(oexpr, ModInst)
        expr = op_subst(instances[1:])

    # A 1st or 2nd order operator has arguments.  However, when passed as
    # an argument to another operator no arguments are specified.  In that
    # case it should not be expanded here.
    if len(oargs) > 0 and iargs == []:
        return iop

    # Check that the arity of the operator is correct
    if len(oargs) != len(iargs):
        print("arity mismatch", lex, "expected:", len(oargs), "got:", len(iargs))
        raise PlusPyError("critical failure")

    # Do a substitution, replacing argument names with argument values
    subs = {}
    for i in range(len(oargs)):
        subs[oargs[i]] = iargs[i]

    x = expr.substitute(subs)
    if isinstance(x, BuiltinExpression):
        return BuiltinExpression(id=x.id, args=x.args, wrapper=x.wrapper, lex=lex, primed=x.primed)
    return x


# This is an expression of the form A!B(b)!C, say
def compile_op_expression(od):
    _primed = False

    # print("COE", od)
    (t0, a0) = od[0]
    assert t0 in [
        "GGeneralIdentifier",
        "GGeneralPrefixOp",
        "GGeneralInfixOp",
        "GGeneralPostfixOp",
    ]
    assert len(a0) == 2

    (t1, a1) = a0
    assert t1 == "Concat"
    assert len(a1) == 2

    # get the list of instances in the prefix
    (operators, instances) = getprefix(a1[0], modstk[-1].operators)

    # get the identifier and arguments
    # print("COE1", a1[1])
    (t2, a2) = a1[1]
    assert t2 in {"Identifier", "Tok"}
    name = a2.lexeme
    (t3, a3) = od[1]
    assert t3 == "Maybe"
    if a3 is None:
        args = []
    else:
        (t4, a4) = a3
        assert t4 == "CommaList"
        args = a4

    cargs = []
    for t, a in args:
        assert t == "GArgument"
        comp = compile_expression(a)
        if comp.primed:
            # primed = True
            pass
        cargs = cargs + [comp]

    # We are now at a point where we have to figure out whether this
    # is the name of an operator or another identifier such as a
    # variable.  If there was a prefix or there are arguments, it must
    # be the name of an operator.  If not, it could be either.
    id = name_lookup(name)
    # print("OE", name, id, cargs, operators.get(id))
    if id and not isinstance(id, OperatorExpression):
        assert instances == []
        if cargs == []:
            return id
        if isinstance(id, ConstantExpression):
            assert id.count == len(cargs)
            return ParameterExpression(id, cargs)
        else:
            assert isinstance(id, ArgumentExpression)
            assert id.nargs == len(cargs)
            return ParameterExpression(id, cargs)
    elif operators.get(name) is None:
        print("unknown identifier", str(a2))
        raise PlusPyError("critical failure")
    else:
        id = operators[name]

    assert isinstance(id, OperatorExpression)
    return op_subst(instances + [(a2, id, cargs)])


def compile_quant_bound_expression(which, qs, ex):
    quantifiers = []
    domains = []
    (t, a) = qs
    assert t == "CommaList"  # one or more quantifiers
    assert len(a) > 0
    for q in a:  # loop through these
        (t2, a2) = q
        assert t2 == "GQuantifierBound"
        domain = compile_expression(a2[1])
        (t3, a3) = a2[0]
        assert t3 in {"CommaList", "Tuple"}
        assert t3 == "CommaList"  # ignore tuples for now
        for t4, a4 in a3:
            assert t4 == "Identifier"
            quantifiers += [BoundvarExpression(a4.lexeme)]
            domains += [domain]

    name_stack.append({bv.id: bv for bv in quantifiers})
    expr = compile_expression(ex)
    name_stack.pop()

    if which == "exists":
        return ExistsExpression(quantifiers=quantifiers, domains=domains, expr=expr, primed=expr.primed)
    if which == "forall":
        return ForallExpression(quantifiers=quantifiers, domains=domains, expr=expr, primed=expr.primed)
    if which == "lambda":
        return LambdaExpression(quantifiers=quantifiers, domains=domains, expr=expr, primed=expr.primed)
    if which == "gen":
        return GenExpression(expr=expr, quantifiers=quantifiers, domains=domains, primed=expr.primed)
    raise ValueError("Invalid quantifier type")


def compile_quant_unbound_expression(which, func):
    quantifiers = []
    (t, a) = func[0]
    assert t == "CommaList"  # one or more quantifiers
    assert len(a) > 0
    for q in a:  # loop through these
        (t2, a2) = q
        assert t2 == "Identifier"
        quantifiers.append(VariableExpression(a2.lexeme))

    name_stack.append({bv.id: bv for bv in quantifiers})
    expr = compile_expression(func[1])
    name_stack.pop()

    if which == "temporal_exists":
        return TemporalExistsExpression(quantifiers=quantifiers, expr=expr, primed=expr.primed)
    if which == "temporal_forall":
        return TemporalForallExpression(quantifiers=quantifiers, expr=expr, primed=expr.primed)
    raise ValueError("Invalid quantifier type")


def compile_expression(ast):
    (t, a) = ast
    if t is False:
        print("compile_expression", a)
        raise ValueError("compile_expression `token` is False")
    elif t == "op":
        return compile_op_expression(a)
    elif t in {"arg-prefix", "arg-infix", "arg-postfix"}:
        return compile_op_expression([a, ("Maybe", None)])
    elif t in {"exists", "forall", "lambda"}:
        return compile_quant_bound_expression(t, a[0], a[1])
    elif t == "gen":
        return compile_quant_bound_expression(t, a[1], a[0])
    elif t in {"temporal_exists", "temporal_forall"}:
        return compile_quant_unbound_expression(t, a)
    elif t in {"GBasicExpression", "parentheses"}:
        return compile_expression(a)
    elif t == "Tuple":
        return TupleExpression().from_ast(a)
    elif t == "set":
        return SetExpression().from_ast(a)
    elif t == "filter":
        return FilterExpression().from_ast(a)
    elif t == "Number":
        return NumberExpression(a)
    elif t == "String":
        return StringExpression(a[1:-1])
    elif t == "Index":
        return IndexExpression().from_ast(a)
    elif t.startswith("Prefix"):
        return OutfixExpression().from_ast(a)
    elif t.startswith("Postfix"):
        (expr, op) = a
        if op.lexeme == "'":
            return PrimeExpression().from_ast(expr)
        else:
            return OutfixExpression().from_ast(a)
    elif t.startswith("Infix"):
        return InfixExpression().from_ast(a)
    elif t == "Cartesian":
        return CartesianExpression().from_ast(a)
    elif t == "choose":
        return ChooseExpression().from_ast(a)
    elif t == "if":
        return IfExpression().from_ast(a)
    elif t == "case":
        return CaseExpression().from_ast(a)
    elif t == "let":
        return LetExpression().from_ast(a)
    elif t == "recordvalue":
        return RecordvalueExpression().from_ast(a)
    elif t == "funcset":
        return FuncsetExpression().from_ast(a)
    elif t == "except":
        return ExceptExpression().from_ast(a)
    elif t == "square":
        return SquareExpression().from_ast(a)
    elif t == "recorddef":
        return RecorddefExpression().from_ast(a)
    elif t in {"wf", "sf"}:
        return FairnessExpression(t, a)
    elif t == "at":
        return name_find("@")
    else:
        logger.error("Can't compile ", ast)
        return None


# handle an "Operator(args) == Expression" definition
def compile_operator_definition(od):
    (t0, a0) = od[0]
    if t0 == "GNonFixLHS":
        assert len(a0) == 2
        (t2, a2) = a0[0]
        assert t2 == "Identifier"
        id = a2.lexeme
        (t3, a3) = a0[1]
        assert t3 == "Maybe"
        if a3 is None:
            args = []
        else:
            (t4, a4) = a3
            assert t4 == "CommaList"
            args = a4

        cargs = []
        for t, a in args:
            assert t == "GOpDecl"
            (t2, a2) = a
            if t2 == "Identifier":
                cargs = cargs + [(a2.lexeme, 0)]
            elif t2 == "paramOp":
                (t3, a3) = a2[0]
                assert t3 == "Identifier"
                (t4, a4) = a2[1]
                assert t4 == "CommaList"
                cargs = cargs + [(a3.lexeme, len(a4))]
            elif t2 == "prefixOp" or t2 == "postfixOp":
                cargs = cargs + [(a2, 1)]
            elif t2 == "infixOp":
                cargs = cargs + [(a2, 2)]
            else:
                raise ValueError("Invalid operator declaration")
    elif t0 == "prefix":
        (t1, a1) = a0
        assert t1 == "Concat"
        (t3, a3) = a1[0]
        assert t3 == "Tok"
        (t4, a4) = a1[1]
        assert t4 == "Identifier"
        id = a3.lexeme
        cargs = [(a4.lexeme, 0)]
    elif t0 == "infix":
        (t1, a1) = a0
        assert t1 == "Concat"
        (t2, a2) = a1[0]
        assert t2 == "Identifier"
        (t3, a3) = a1[1]
        assert t3 == "Tok"
        (t4, a4) = a1[2]
        assert t4 == "Identifier"
        id = a3.lexeme
        cargs = [(a2.lexeme, 0), (a4.lexeme, 0)]
    elif t0 == "postfix":
        (t1, a1) = a0
        assert t1 == "Concat"
        (t2, a2) = a1[0]
        assert t2 == "Identifier"
        (t3, a3) = a1[1]
        assert t3 == "Tok"
        id = a3.lexeme
        cargs = [(a2.lexeme, 0)]
    else:
        logger.error(f"compile_operator_definition {t0} {a0}")
        raise ValueError("Invalid operator definition")

    # print("OD", modstk[-1].name, id)
    args = [ArgumentExpression(a, n) for (a, n) in cargs]
    name_stack.append({a.id: a for a in args})
    ce = compile_expression(od[1])
    name_stack.pop()

    return (id, args, ce)


# handle a "Function[args] == Expression" definition.  Define as
#   f[x \in D] == e  ==>   f == CHOOSE f: f = [x \ D: e]
def compile_function_definition(od):
    (t0, a0) = od[0]
    assert t0 == "Identifier"
    id = a0.lexeme
    bve = BoundvarExpression(id)
    name_stack.append({id: bve})
    f = compile_quant_bound_expression("lambda", od[1], od[2])
    name_stack.pop()
    (op, file, column, first) = a0
    infix = InfixExpression(op=("=", file, column, first), lhs=bve, rhs=f)
    c = ChooseExpression(id=bve, expr=infix)
    return (id, [], c)


class Expression:
    def __init__(self):
        self.primed = None  # set if this expression is primed

    def __repr__(self):
        return self.__str__()

    def run_init(self, containers, boundedvars):
        logger.error("run_init", self)
        raise NotImplementedError("run_init not implemented")

    def eval(self, containers, boundedvars):
        logger.error("Eval: ", self)
        raise NotImplementedError("eval not implemented")

    def apply(self, containers, boundedvars, fargs):
        v = self.eval(containers, boundedvars)
        if v is None:
            logger.info(f"Default apply {self} {fargs}")
        assert v is not None
        return funceval(v, fargs)


# A built-in expression
class BuiltinExpression(Expression):
    def __init__(self, id=None, args=None, wrapper=None, lex=None, primed=False):
        self.id = id
        self.args = args
        self.wrapper = wrapper
        self.lex = lex
        self.primed = primed

    def __str__(self):
        return (
            "Builtin("
            + self.id
            + ", "
            + str(self.args)
            + ", "
            + str(self.wrapper)
            + ", "
            + str(self.lex)
            + ")"
        )

    def substitute(self, subs):
        args = [x.substitute(subs) for x in self.args]
        return BuiltinExpression(
            id=self.id,
            args=args,
            wrapper=self.wrapper,
            lex=self.lex,
            primed=self.primed,
        )

    def eval(self, containers, boundedvars):
        # print("BI eval", self)
        args = [arg.eval(containers, boundedvars) for arg in self.args]
        try:
            return self.wrapper(self.id, args)
        except Exception as e:
            print("Evaluating", str(self.lex), "failed")
            print(e)
            print(traceback.format_exc())
            raise PlusPyError("critical failure") from e


# The simplest of expressions is just a value
class ValueExpression(Expression):
    def __init__(self, value=None, primed=False):
        self.value = value
        self.primed = primed

    def __str__(self):
        return "Value(" + str(self.value) + ")"

    def substitute(self, subs):
        return self

    def eval(self, containers, boundedvars):
        return self.value


name_stack[-1]["FALSE"] = ValueExpression(False)
name_stack[-1]["TRUE"] = ValueExpression(True)


# Another simple one is a variable expression
class VariableExpression(Expression):
    def __init__(self, id=None, primed=False):
        self.id = id
        self.primed = primed

    def __str__(self):
        return "Variable(" + str(self.id) + ")"

    def substitute(self, subs):
        if subs.get(self) is None:
            return self
        else:
            global initializing
            if initializing:
                return PrimeExpression(expr=subs[self])
            else:
                return subs[self]

    def eval(self, containers, boundedvars):
        print("Error: variable", self.id, "not realized", containers, boundedvars)
        raise PlusPyError("critical failure")


# Another simple one is a constant expression
class ConstantExpression(Expression):
    def __init__(self, id=None, count=0, primed=False):
        self.id = id
        self.count = count
        self.primed = primed

    def __str__(self):
        return "Constant(" + self.id + ", " + str(self.count) + ")"

    def substitute(self, subs):
        if subs.get(self) is None:
            return self
        else:
            return subs[self]

    def eval(self, containers, boundedvars):
        print("Error: constant", self.id, "does not have a value")
        raise PlusPyError("critical failure")


# Another simple one is a bounded variable (in \E, lambdas, etc.)
# The values are in the "boundedvars" dictionary
class BoundvarExpression(Expression):
    def __init__(self, id=None, primed=False):
        self.id = id
        self.primed = primed

        global bv_counter
        bv_counter += 1
        self.uid = bv_counter

    def __str__(self):
        return "Boundvar(" + str(self.id) + ", " + str(self.uid) + ")"

    def substitute(self, subs):
        return self

    def eval(self, containers, boundedvars):
        expr = boundedvars[self]
        assert isinstance(expr, ValueExpression)
        return expr.eval(containers, boundedvars)

    def apply(self, containers, boundedvars, fargs):
        expr = boundedvars[self]
        return expr.apply(containers, boundedvars, fargs)


# An "argument" is the usage of an argument to an operator definition
# inside its body.  It itself may have arguments.  Needs to be substituted
# before evaluation
class ArgumentExpression(Expression):
    def __init__(self, id=None, nargs=0, primed=False):
        self.id = id
        self.nargs = nargs
        self.primed = primed

    def __str__(self):
        return "Argument(" + str(self.id) + ", " + str(self.nargs) + ")"

    def substitute(self, subs):
        if subs.get(self) is None:
            return self
        else:
            return subs[self]

    def eval(self, containers, boundedvars):
        logger.error(f"Argument {self.id} not realized {self.nargs} {containers} {boundedvars}")
        raise ValueError("Argument not realized")


# This is like an ArgumentExpression with arguments of its own (i.e., an
# argument of a 2nd order operator, but with its arguments instantiated
# It still needs to be substituted with an actual operator before evaluation
class ParameterExpression(Expression):
    def __init__(self, argument=None, args=None, primed=False):
        self.argument = argument
        self.args = args
        self.primed = primed

    def __str__(self):
        return "Parameter(" + str(self.argument) + ", " + str(self.args) + ")"

    def substitute(self, subs):
        if subs.get(self.argument):
            op = subs.get(self.argument)
            if isinstance(op, OperatorExpression):
                assert isinstance(op, OperatorExpression)
                assert len(self.args) == len(op.args)
                s = subs.copy()
                for i in range(len(self.args)):
                    s[op.args[i]] = self.args[i].substitute(s)
                return op.expr.substitute(s)
            else:
                assert isinstance(op, ArgumentExpression)
                assert len(self.args) == op.nargs
                # print("ZZZ", self, op, subs)
                return ParameterExpression(argument=op, args=self.args, primed=self.primed)
        else:
            args = [a.substitute(subs) for a in self.args]
            return ParameterExpression(argument=self.argument, args=args, primed=self.primed)

    def eval(self, containers, boundedvars):
        logger.error(f"Parameter {self.argument} not realized")
        raise ValueError("Parameter not realized")


class OperatorExpression(Expression):
    def __init__(self, id=None, args=None, expr=None, primed=False):
        self.id = id
        self.args = args
        self.expr = expr
        self.primed = primed

    def __str__(self):
        return "OperatorExpression(" + self.id + ", " + str(self.args) + ")"
        # + ", " + self.expr.__str__() \

    def substitute(self, subs):
        return OperatorExpression(
            id=self.id,
            args=self.args,
            expr=self.expr.substitute(subs),
            primed=self.primed,
        )

    def eval(self, containers, boundedvars):
        # print("operator", self, "invoked without arguments")
        return self


# Another simple one is a container expression, which holds a value for a variable
# for both the previous state and the next state
class ContainerExpression(Expression):
    def __init__(self, var=None, primed=False):
        self.var = var
        self.primed = primed
        self.prev = None
        self.next = None

    def __str__(self):
        return (
            "Container(" + self.var.id + ", " + str(convert(self.prev)) + ", " + str(convert(self.next)) + ")"
        )

    def substitute(self, subs):
        return self

    def eval(self, containers, boundedvars):
        if self.prev is None:
            print("null container", self)
        assert self.prev is not None
        return self.prev


class SquareExpression(Expression):
    def __init__(self, lhs=None, rhs=None, primed=False):
        self.lhs = lhs
        self.rhs = rhs
        self.primed = primed

    def from_ast(self, exprs):
        assert len(exprs) == 2
        self.lhs = compile_expression(exprs[0])
        self.rhs = compile_expression(exprs[1])
        assert not self.rhs.primed
        self.primed = self.lhs.primed
        return self

    def __str__(self):
        return "Square(" + self.lhs.__str__() + ", " + self.rhs.__str__() + ")"

    def substitute(self, subs):
        lhs = self.lhs.substitute(subs)
        rhs = self.rhs.substitute(subs)
        return SquareExpression(lhs=lhs, rhs=rhs, primed=self.primed)

    def eval(self, containers, boundedvars):
        return self.lhs.eval(containers, boundedvars)


class FairnessExpression(Expression):
    def __init__(self, t, a):
        self.type = t
        (t0, a0) = a[0]
        if t0 == "Identifier":
            self.lhs = VariableExpression(id=a0.lexeme)
        else:
            self.lhs = compile_expression(a[0])
        self.rhs = compile_expression(a[1])
        assert not self.lhs.primed
        self.primed = self.rhs.primed

    def __str__(self):
        return f"FAIRNESS({self.type}, lhs={str(self.lhs)}, rhs={str(self.rhs)})"

    def substitute(self, subs):
        return self


class LambdaExpression(Expression):
    def __init__(self, quantifiers=None, domains=None, expr=None, primed=False):
        self.quantifiers = quantifiers
        self.domains = domains
        self.expr = expr
        self.primed = primed

    def __str__(self):
        return "Lambda(" + str(self.quantifiers) + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        domains = [expr.substitute(subs) for expr in self.domains]
        expr = self.expr.substitute(subs)
        return LambdaExpression(quantifiers=self.quantifiers, domains=domains, expr=expr, primed=self.primed)

    def enumerate(self, containers, domains, lst, result, boundedvars):
        if domains == []:
            if len(lst) == 1:
                result[lst[0]] = self.expr.eval(containers, boundedvars)
            else:
                result[tuple(lst)] = self.expr.eval(containers, boundedvars)
        else:
            (var, domain) = domains[0]
            if domain is False:
                print("Error: possibly trying to evaluate Nat")
                raise PlusPyError("critical failure")
            domain = sorted(domain, key=lambda x: key(x))
            for val in domain:
                boundedvars[var] = ValueExpression(val)
                self.enumerate(containers, domains[1:], lst + [val], result, boundedvars)

    def eval(self, containers, boundedvars):
        domains = []
        for i in range(len(self.quantifiers)):
            domains += [(self.quantifiers[i], self.domains[i].eval(containers, boundedvars))]
        result = {}
        self.enumerate(containers, domains, [], result, boundedvars.copy())
        return simplify(FrozenDict(result))

    def apply(self, containers, boundedvars, fargs):
        assert len(self.quantifiers) == len(fargs)
        bv = boundedvars.copy()
        for i in range(len(fargs)):
            var = self.quantifiers[i]
            bv[var] = ValueExpression(fargs[i])
        return self.expr.eval(containers, bv)


class ExistsExpression(Expression):
    def __init__(self, quantifiers=None, domains=None, expr=None, primed=False):
        self.quantifiers = quantifiers
        self.domains = domains
        self.expr = expr
        self.primed = primed

    def __str__(self):
        return "Exists(" + str(self.quantifiers) + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        domains = [expr.substitute(subs) for expr in self.domains]
        expr = self.expr.substitute(subs)
        return ExistsExpression(quantifiers=self.quantifiers, domains=domains, expr=expr, primed=self.primed)

    def enumerate(self, containers, domains, boundedvars):
        if domains == []:
            return self.expr.eval(containers, boundedvars)
        (var, domain) = domains[0]

        # Pseudo-randomized SAT solving...
        domain = sorted(domain, key=lambda x: key(x))
        domain = random.sample(list(domain), len(domain))

        # Copy next state in case need to restore
        output_copy = run_global_vars.IO_outputs.copy()
        waitset_copy = run_global_vars.waitset.copy()
        signalset_copy = run_global_vars.signalset.copy()
        copy = {}
        for k, v in containers.items():
            copy[k] = v.next
        for val in domain:
            boundedvars[var] = ValueExpression(val)
            if self.enumerate(containers, domains[1:], boundedvars):
                return True
            # restore state before trying next
            for k, v in copy.items():
                containers[k].next = v
            run_global_vars.IO_outputs = output_copy
            run_global_vars.waitset = waitset_copy
            run_global_vars.signalset = signalset_copy
        return False

    def eval(self, containers, boundedvars):
        domains = []
        for i in range(len(self.quantifiers)):
            domains += [(self.quantifiers[i], self.domains[i].eval(containers, boundedvars))]
        return self.enumerate(containers, domains, boundedvars.copy())


class ForallExpression(Expression):
    def __init__(self, quantifiers=None, domains=None, expr=None, primed=False):
        self.quantifiers = quantifiers
        self.domains = domains
        self.expr = expr
        self.primed = primed

    def __str__(self):
        return "Forall(" + str(self.quantifiers) + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        domains = [expr.substitute(subs) for expr in self.domains]
        expr = self.expr.substitute(subs)
        return ForallExpression(quantifiers=self.quantifiers, domains=domains, expr=expr, primed=self.primed)

    def enumerate(self, containers, domains, boundedvars):
        if domains == []:
            return self.expr.eval(containers, boundedvars)
        (var, domain) = domains[0]
        domain = sorted(domain, key=lambda x: key(x))
        for val in domain:
            boundedvars[var] = ValueExpression(val)
            if not self.enumerate(containers, domains[1:], boundedvars):
                return False
        return True

    # TODO.  This may not work for primed expressions currently
    def eval(self, containers, boundedvars):
        domains = []
        for i in range(len(self.quantifiers)):
            domains += [(self.quantifiers[i], self.domains[i].eval(containers, boundedvars))]
        return self.enumerate(containers, domains, boundedvars.copy())


class GenExpression(Expression):
    def __init__(self, expr=None, quantifiers=None, domains=None, primed=False):
        self.expr = expr
        self.quantifiers = quantifiers
        self.domains = domains
        self.primed = primed

    def __str__(self):
        return f"Gen({str(self.expr)}, {str(self.quantifiers)})"

    def substitute(self, subs):
        domains = [expr.substitute(subs) for expr in self.domains]
        expr = self.expr.substitute(subs)
        return GenExpression(expr=expr, quantifiers=self.quantifiers, domains=domains, primed=self.primed)

    def enumerate(self, containers, domains, boundedvars, result):
        if domains == []:
            result.append(self.expr.eval(containers, boundedvars))
        else:
            (var, domain) = domains[0]
            domain = sorted(domain, key=lambda x: key(x))
            for val in domain:
                boundedvars[var] = ValueExpression(val)
                self.enumerate(containers, domains[1:], boundedvars, result)

    def eval(self, containers, boundedvars):
        domains = []
        for i in range(len(self.quantifiers)):
            domains += [(self.quantifiers[i], self.domains[i].eval(containers, boundedvars))]
        result = []
        self.enumerate(containers, domains, boundedvars.copy(), result)
        return frozenset(result)


class TemporalExistsExpression(Expression):
    def __init__(self, quantifiers=None, expr=None, containers=None, primed=False):
        self.quantifiers = quantifiers
        self.expr = expr
        self.containers = containers
        self.primed = self.expr.primed

    def __str__(self):
        return "TempExists(" + str(self.quantifiers) + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        global initializing
        if initializing:
            containers = subs.copy()
            for id in self.quantifiers:
                containers[id] = ContainerExpression(var=id)
            return TemporalExistsExpression(
                quantifiers=self.quantifiers,
                expr=self.expr.substitute(containers),
                containers=containers,
                primed=self.primed,
            )
        else:
            return TemporalExistsExpression(
                quantifiers=self.quantifiers,
                expr=self.expr.substitute(subs),
                containers=self.containers,
                primed=self.primed,
            )

    def eval(self, containers, boundedvars):
        return self.expr.eval(self.containers, boundedvars)


class TemporalForallExpression(Expression):
    def __init__(self, quantifiers=None, expr=None, containers=None, primed=False):
        self.quantifiers = quantifiers
        self.expr = expr
        self.containers = containers
        self.primed = self.expr.primed

    def __str__(self):
        return "TempForAll(" + str(self.quantifiers) + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        raise NotImplementedError("substitute not implemented")

    def eval(self, *args, **kwargs):
        raise NotImplementedError("eval not implemented")


class RecorddefExpression(Expression):
    def __init__(self, kvs=None, primed=False):
        self.kvs = kvs
        self.primed = primed

    def from_ast(self, ast):
        (t, a) = ast
        assert t == "CommaList"
        self.kvs = dict()
        self.primed = False
        for t2, a2 in a:
            assert t2 == "Concat"
            assert len(a2) == 3
            (t3, a3) = a2[0]
            assert t3 == "Name"
            expr = compile_expression(a2[2])
            self.kvs[a3.lexeme] = expr
            self.primed = self.primed or expr.primed
        return self

    def __str__(self):
        result = ""
        for k, e in self.kvs.items():
            if result != "":
                result += ", "
            result += str(k) + ": " + e.__str__()
        return "Recorddef(" + result + ")"

    def substitute(self, subs):
        kvs = {k: v.substitute(subs) for (k, v) in self.kvs.items()}
        return RecorddefExpression(kvs=kvs, primed=self.primed)

    def expand(self, keys, record, result, containers, boundedvars):
        if keys == []:
            result.append(simplify(FrozenDict(record.copy())))
        else:
            k = keys[0]
            v = self.kvs[k]
            r = v.eval(containers, boundedvars)
            r = sorted(r, key=lambda x: key(x))
            for e in r:
                record[k] = e
                self.expand(keys[1:], record, result, containers, boundedvars)

    def eval(self, containers, boundedvars):
        keys = list(self.kvs.keys())
        keys = sorted(keys, key=lambda x: key(x))
        result = []
        self.expand(keys, {}, result, containers, boundedvars)
        return frozenset(result)


class RecordvalueExpression(Expression):
    def __init__(self, kvs=None, primed=False):
        self.kvs = kvs
        self.primed = primed

    def from_ast(self, ast):
        (t, a) = ast
        assert t == "CommaList"
        self.kvs = dict()
        self.primed = False
        for t2, a2 in a:
            assert t2 == "Concat"
            assert len(a2) == 3
            (t3, a3) = a2[0]
            assert t3 == "Name"
            expr = compile_expression(a2[2])
            self.kvs[a3.lexeme] = expr
            self.primed = self.primed or expr.primed
        return self

    def __str__(self):
        result = ""
        for k, e in self.kvs.items():
            if result != "":
                result += ", "
            result += str(k) + ": " + e.__str__()
        return "Recordvalue(" + result + ")"

    def substitute(self, subs):
        kvs = {k: v.substitute(subs) for (k, v) in self.kvs.items()}
        return RecordvalueExpression(kvs=kvs, primed=self.primed)

    def eval(self, containers, boundedvars):
        kvs = dict()
        keys = self.kvs.keys()
        for k in sorted(keys, key=lambda x: key(x)):
            kvs[k] = self.kvs[k].eval(containers, boundedvars)
        return simplify(FrozenDict(kvs))


class FuncsetExpression(Expression):
    def __init__(self, lhs=None, rhs=None, primed=False):
        self.lhs = lhs
        self.rhs = rhs
        self.primed = primed

    def from_ast(self, exprs):
        assert len(exprs) == 2
        self.lhs = compile_expression(exprs[0])
        self.rhs = compile_expression(exprs[1])
        self.primed = self.lhs.primed or self.rhs.primed
        return self

    def __str__(self):
        return "FuncSet(" + self.lhs.__str__() + ", " + self.rhs.__str__() + ")"

    def substitute(self, subs):
        return FuncsetExpression(
            lhs=self.lhs.substitute(subs),
            rhs=self.rhs.substitute(subs),
            primed=self.primed,
        )

    def enumerate(self, lhs, rhs, record, result):
        if lhs == []:
            result.append(simplify(FrozenDict(record.copy())))
        else:
            for y in rhs:
                record[lhs[0]] = y
                self.enumerate(lhs[1:], rhs, record, result)

    def eval(self, containers, boundedvars):
        lhs = self.lhs.eval(containers, boundedvars)
        rhs = self.rhs.eval(containers, boundedvars)
        result = []
        self.enumerate(list(lhs), list(rhs), {}, result)
        return frozenset(result)


class ExceptExpression(Expression):
    def __init__(self, lhs=None, rhs=None, at=None, primed=False):
        self.lhs = lhs
        self.rhs = rhs
        self.at = at
        self.primed = primed

    def from_ast(self, exc):
        assert len(exc) == 2
        self.lhs = compile_expression(exc[0])
        self.at = BoundvarExpression("@")
        self.primed = self.lhs.primed
        (t, a) = exc[1]
        assert t == "GExcept"
        assert len(a) > 0
        self.rhs = []
        for lst, expr in a:
            args = []
            for arg in lst:
                (t2, a2) = arg
                assert t2 in {"elist", "efield"}
                (t3, a3) = a2
                if t2 == "elist":
                    assert t3 == "CommaList"
                    assert len(a3) > 0
                    indices = []
                    for e in a3:
                        ce = compile_expression(e)
                        if ce.primed:
                            self.primed = True
                        indices += [ce]
                    args += [indices]
                else:
                    assert t3 == "Name"
                    args += [[StringExpression(a3.lexeme)]]
            name_stack.append({"@": self.at})
            cexpr = compile_expression(expr)
            name_stack.pop()
            if cexpr.primed:
                self.primed = True
            self.rhs += [(args, cexpr)]
        return self

    def __str__(self):
        result = ""
        for args, expr in self.rhs:
            ind = ""
            for a in args:
                if ind != "":
                    ind += ", "
                pos = ""
                for x in a:
                    if pos != "":
                        pos += ", "
                    pos += x.__str__()
                ind += "[" + pos + "]"
            if result != "":
                result += ", "
            result += "Replace(" + ind + ", " + expr.__str__() + ")"
        return "Except(" + self.lhs.__str__() + ", [" + result + "])"

    def substitute(self, subs):
        lhs = self.lhs.substitute(subs)
        rhs = []
        for args, expr in self.rhs:
            ind = []
            for a in args:
                pos = []
                for x in a:
                    pos += [x.substitute(subs)]
                ind += [pos]
            rhs += [(ind, expr.substitute(subs))]
        return ExceptExpression(lhs=lhs, rhs=rhs, at=self.at, primed=self.primed)

    def eval(self, containers, boundedvars):
        lhs = self.lhs.eval(containers, boundedvars)
        if isinstance(lhs, str) or isinstance(lhs, tuple):
            kvs = {(i + 1): lhs[i] for i in range(len(lhs))}
        else:
            assert isinstance(lhs, FrozenDict)
            kvs = lhs.d.copy()

        # Evaluate the exceptions
        for iargs, iexpr in self.rhs:
            assert len(iargs) == 1  # TODO doesn't handle ![][]...
            a = iargs[0]
            vals = [arg.eval(containers, boundedvars) for arg in a]
            new_bvs = boundedvars.copy()
            old = funceval(lhs, vals)
            new_bvs[self.at] = ValueExpression(old)
            new = iexpr.eval(containers, new_bvs)
            if len(vals) == 1:
                kvs[vals[0]] = new
            else:
                kvs[tuple(vals)] = new
        return simplify(FrozenDict(kvs))


class PrimeExpression(Expression):
    def __init__(self, expr=None, primed=True):
        self.expr = expr
        assert primed
        self.primed = primed

    def from_ast(self, expr):
        self.expr = compile_expression(expr)
        assert self.expr.primed is False
        return self

    def __str__(self):
        return "Prime(" + self.expr.__str__() + ")"

    def substitute(self, subs):
        return PrimeExpression(expr=self.expr.substitute(subs))

    def eval(self, containers, boundedvars):
        assert isinstance(self.expr, ContainerExpression)
        assert self.expr.next is not None
        return self.expr.next


class OutfixExpression(Expression):
    def __init__(self, op=None, expr=None, primed=False):
        self.op = op
        assert not isinstance(expr, tuple)
        self.expr = expr
        self.primed = primed

    def from_ast(self, prefix):
        (op, expr) = prefix
        lex = op.lexeme
        self.op = "-." if lex == "-" else lex

        mod = modstk[-1]
        if self.op in mod.operators:
            id = mod.operators[self.op]
            assert isinstance(id, OperatorExpression)
            assert len(id.args) == 1
            args = [compile_expression(expr)]
            return op_subst([(op, id, args)])

        self.expr = compile_expression(expr)
        assert not isinstance(self.expr, tuple)

        if self.op == "-." and isinstance(self.expr, NumberExpression):
            return NumberExpression(-self.expr.number)
        self.primed = self.expr.primed
        return self

    def __str__(self):
        return 'Outfix("' + self.op + '", ' + self.expr.__str__() + ")"

    def substitute(self, subs):
        # TODO.  Perhaps operator itself must be substituted?
        assert subs.get(self.op) is None

        global initializing
        if self.op == "[]" and initializing:
            initializing = False
            expr = self.expr.substitute(subs)
            initializing = True
            return OutfixExpression(op=self.op, expr=expr, primed=self.primed)

        return OutfixExpression(op=self.op, expr=self.expr.substitute(subs), primed=self.primed)

    def always(self, containers, boundedvars):
        assert isinstance(self.expr, SquareExpression)

        ok = True
        for k, v in containers.items():
            if v.next is None:
                print("always: UNASSIGNED", k)
                ok = False
        assert ok

        op = self.expr.lhs
        tries = 0
        i = 0
        while True:
            if run_global_vars.maxcount is not None and i >= run_global_vars.maxcount:
                return None
            for c in containers.values():
                c.prev = c.next
                c.next = None
            r = op.eval(containers, boundedvars)
            if r:
                changed = False
                for c in containers.values():
                    if c.next != c.prev:
                        changed = True
                        break
                if not changed:
                    break
                tries = 0
            else:
                for c in containers.values():
                    c.next = c.prev
                tries += 1
                if tries % 100 == 0:
                    print("always: try again", tries)
                    run_global_vars.cond.wait(0.2)
            i += 1

    def unchanged(self, expr):
        if isinstance(expr, TupleExpression):
            for x in expr.exprs:
                r = self.unchanged(x)
                if not r:
                    return False
        else:
            assert isinstance(expr, ContainerExpression)
            if expr.next is not None and expr.next != expr.prev:
                return False
            expr.next = expr.prev
        return True

    def eval(self, containers, boundedvars):
        if self.op == "UNCHANGED":
            return self.unchanged(self.expr)

        if self.op == "[]":
            return self.always(containers, boundedvars)

        _v = self.expr.eval(containers, boundedvars)

        logger.error(f"Outfix operator {self.op} not defined")
        raise ValueError("Outfix operator not defined")


class ChooseExpression(Expression):
    def __init__(self, id=None, domain=None, expr=None, primed=False):
        self.id = id
        self.domain = domain
        self.expr = expr
        self.primed = primed

    def from_ast(self, expr):
        assert len(expr) == 3
        (t, a) = expr[0]
        assert t == "Identifier"
        self.id = BoundvarExpression(a.lexeme)
        (t1, a1) = expr[1]
        assert t1 == "Maybe"
        self.domain = None if a1 is None else compile_expression(a1)

        name_stack.append({self.id.id: self.id})
        self.expr = compile_expression(expr[2])
        name_stack.pop()
        self.primed = False
        return self

    def __str__(self):
        return "Choose(" + str(self.id) + ", " + self.domain.__str__() + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        return ChooseExpression(
            id=self.id,
            domain=None if self.domain is None else self.domain.substitute(subs),
            expr=self.expr.substitute(subs),
            primed=self.primed,
        )

    def eval(self, containers, boundedvars):
        new_bv = boundedvars.copy()
        if self.domain is None:
            lexeme = self.expr.op[0]
            if (
                isinstance(self.expr, InfixExpression)
                and lexeme in {"=", "\\in", "\\notin"}
                and isinstance(self.expr.lhs, BoundvarExpression)
                and self.expr.lhs == self.id
            ):
                match lexeme:
                    case "=":
                        func = self.expr.rhs
                        new_bv[self.id] = func
                        return func.eval(containers, new_bv)
                    case "\\in":
                        func = self.expr.rhs
                        new_bv[self.id] = func
                        s = sorted(func.eval(containers, new_bv), key=lambda x: key(x))
                        return s[0]
                    case "\\notin":
                        # CHOOSE of same expression should return same value...
                        x = val_to_string(self.expr.rhs)
                        return Nonce(x.__hash__())
                    case _:
                        raise ValueError("ChooseExpression: unknown operator")
            elif isinstance(self.expr, ValueExpression) and isinstance(self.expr.value, bool):
                return Nonce(self.expr.value.__hash__())
        else:
            domain = sorted(self.domain.eval(containers, boundedvars), key=lambda x: key(x))
            for x in domain:
                new_bv[self.id] = ValueExpression(x)
                r = self.expr.eval(containers, new_bv)
                if r:
                    return x
        logger.error(f"CHOOSE {self}")
        raise ValueError("ChooseExpression: no value found")

    def apply(self, containers, boundedvars, fargs):
        new_bv = boundedvars.copy()
        if (
            self.domain is None
            and isinstance(self.expr, InfixExpression)
            and self.expr.op[0] == "="
            and isinstance(self.expr.lhs, BoundvarExpression)
            and self.expr.lhs == self.id
        ):
            func = self.expr.rhs
            new_bv[self.id] = func
            return func.apply(containers, new_bv, fargs)
        else:
            v = self.eval(containers, boundedvars)
            return funceval(v, fargs)


# TODO.  Can potentiallly get rid of this in favor of CaseExpression
class IfExpression(Expression):
    def __init__(self, cond=None, ifexpr=None, elseexpr=None, primed=False):
        self.cond = cond
        self.ifexpr = ifexpr
        self.elseexpr = elseexpr
        self.primed = primed

    def from_ast(self, expr):
        assert len(expr) == 3
        self.cond = compile_expression(expr[0])
        self.ifexpr = compile_expression(expr[1])
        self.elseexpr = compile_expression(expr[2])
        self.primed = self.cond.primed or self.ifexpr.primed or self.elseexpr.primed
        return self

    def __str__(self):
        return (
            "If(" + self.cond.__str__() + ", " + self.ifexpr.__str__() + ", " + self.elseexpr.__str__() + ")"
        )

    def substitute(self, subs):
        return IfExpression(
            cond=self.cond.substitute(subs),
            ifexpr=self.ifexpr.substitute(subs),
            elseexpr=self.elseexpr.substitute(subs),
            primed=self.primed,
        )

    def eval(self, containers, boundedvars):
        cond = self.cond.eval(containers, boundedvars)
        if cond:
            return self.ifexpr.eval(containers, boundedvars)
        else:
            return self.elseexpr.eval(containers, boundedvars)


class CaseExpression(Expression):
    def __init__(self, cases=None, other=None, primed=False):
        self.cases = cases
        self.other = other
        self.primed = primed

    def from_ast(self, expr):
        (t0, a0) = expr[0]
        assert t0 == "SeparatorList"
        (t1, a1) = expr[1]
        assert t1 == "Maybe"

        self.primed = False
        self.cases = []
        for t2, a2 in a0:
            assert t2 == "Concat"
            cond = compile_expression(a2[0])
            val = compile_expression(a2[2])
            self.cases += [(cond, val)]
            if cond.primed or val.primed:
                self.primed = True

        if a1 is None:
            self.other = None
        else:
            self.other = compile_expression(a1)
            self.primed = self.primed or self.other.primed

        return self

    def __str__(self):
        result = ""
        for c, e in self.cases:
            if result != "":
                result += " [] "
            result += c.__str__() + " -> " + e.__str__()
        if self.other is not None:
            result += " [] OTHER -> " + self.other.__str__()
        return "Case(" + result + ")"

    def substitute(self, subs):
        cases = [(cond.substitute(subs), expr.substitute(subs)) for (cond, expr) in self.cases]
        other = None if self.other is None else self.other.substitute(subs)
        return CaseExpression(cases=cases, other=other, primed=self.primed)

    def eval(self, containers, boundedvars):
        cases = random.sample(self.cases, len(self.cases))
        for c, e in cases:
            r = c.eval(containers, boundedvars)
            if r:
                return e.eval(containers, boundedvars)
        assert self.other is not None
        return self.other.eval(containers, boundedvars)


class LetExpression(Expression):
    def __init__(self, mod=None, expr=None, primed=False):
        self.mod = mod
        self.expr = expr
        self.primed = primed

    def from_ast(self, expr):
        assert len(expr) == 2
        (t, a) = expr[0]
        assert t == "AtLeast1"

        # LET is treated like a mini-module
        self.mod = Module()
        mod = modstk[-1]
        self.mod.variables = mod.variables.copy()
        self.mod.constants = mod.constants.copy()
        self.mod.operators = mod.operators.copy()
        self.mod.wrappers = mod.wrappers.copy()
        self.mod.globals = mod.globals.copy()
        modstk.append(self.mod)
        ops = {}
        name_stack.append(ops)
        for d in a:
            (t1, a1) = d
            if t1 == "GOperatorDefinition":
                (id, args, e) = compile_operator_definition(a1)
            else:
                assert t1 == "GFunctionDefinition"  # deal with ModDef later
                (id, args, e) = compile_function_definition(a1)
            od = OperatorExpression(id, args, e)
            self.mod.operators[id] = od
            self.mod.globals.add(id)
            ops[id] = od
        self.expr = compile_expression(expr[1])
        name_stack.pop()
        modstk.pop()

        return self.expr  # make "LET" disappear

    def __str__(self):
        if False:
            return "Let(" + str(self.mod.operators) + ", " + self.expr.__str__() + ")"
        else:
            return "Let(" + self.expr.__str__() + ")"

    def substitute(self, subs):
        # return LetExpression(mod=self.mod,
        #     expr=self.expr.substitute(subs),
        #     primed=self.primed)
        raise NotImplementedError("LetExpression: substitute not implemented")

    def eval(self, containers, boundedvars):
        raise NotImplementedError("LetExpression: eval not implemented")


# Cartesian product
class CartesianExpression(Expression):
    def __init__(self, exprs=None, primed=False):
        self.exprs = exprs
        self.primed = primed

    def from_ast(self, cart):
        self.exprs = [compile_expression(x) for x in cart]
        self.primed = any(x.primed for x in self.exprs)
        return self

    def __str__(self):
        result = ""
        for x in self.exprs:
            if result != "":
                result += ", "
            result += x.__str__()
        return "Cartesian(" + result + ")"

    def substitute(self, subs):
        exprs = [x.substitute(subs) for x in self.exprs]
        return CartesianExpression(exprs=exprs, primed=self.primed)

    def enumerate(self, exprs, tup, result):
        if exprs == []:
            result.append(tuple(tup))
        else:
            for x in exprs[0]:
                self.enumerate(exprs[1:], tup + [x], result)

    def eval(self, containers, boundedvars):
        exprs = [x.eval(containers, boundedvars) for x in self.exprs]
        result = []
        self.enumerate(exprs, [], result)
        return frozenset(result)


class InfixExpression(Expression):
    def __init__(self, op=None, lhs=None, rhs=None, primed=False):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
        self.primed = primed

    def from_ast(self, infix):
        (op, lhs, rhs) = infix

        lex = op.lexeme
        lt = compile_expression(lhs)
        rt = compile_expression(rhs)
        mod = modstk[-1]

        if lex in mod.operators:
            id = mod.operators[lex]
            assert isinstance(id, OperatorExpression)
            assert len(id.args) == 2
            return op_subst([(op, id, [lt, rt])])

        self.op = op
        self.lhs = lt
        self.rhs = rt
        self.primed = self.lhs.primed or self.rhs.primed
        return self

    def __str__(self):
        return 'Infix("' + str(self.op) + '", ' + self.lhs.__str__() + ", " + self.rhs.__str__() + ")"

    def substitute(self, subs):
        return InfixExpression(
            op=self.op,
            lhs=self.lhs.substitute(subs),
            rhs=self.rhs.substitute(subs),
            primed=self.primed,
        )

    def eval(self, containers, boundedvars):
        lex = self.op.lexeme

        # One special case is if the expression is of the form x' = ...
        # when x' is not assigned a value in next.  In that case we set
        # x' to ...
        if isinstance(self.lhs, PrimeExpression):
            var = self.lhs.expr
            assert isinstance(var, ContainerExpression)
            if var.next is None:
                val = self.rhs.eval(containers, boundedvars)
                if val is None:
                    print("XXX", self.rhs)
                assert val is not None
                if lex == "=":
                    var.next = val
                    # print("ASSIGN", var.var, containers)
                    return True
                elif lex == "\\in":
                    lst = list(val)
                    r = random.randrange(len(lst))
                    var.next = lst[r]
                    return True

        # Copy next state in case need to restore after OR operation
        # with FALSE left hand side.  Also, randomize lhs/rhs
        # evaluation
        if lex == "\\/":
            output_copy = run_global_vars.IO_outputs.copy()
            waitset_copy = run_global_vars.waitset.copy()
            signalset_copy = run_global_vars.signalset.copy()
            copy = {}
            for k, v in containers.items():
                copy[k] = v.next
            r = random.randrange(2)
            if r == 0:
                lhs = self.lhs.eval(containers, boundedvars)
            else:
                assert r == 1
                rhs = self.rhs.eval(containers, boundedvars)
        else:
            r = 0
            lhs = self.lhs.eval(containers, boundedvars)

        if lex == "\\/":
            if (r == 0) and lhs:
                return lhs
            if (r == 1) and rhs:
                return rhs
            # restore and evaluate right hand side
            for k, v in copy.items():
                containers[k].next = v
            run_global_vars.IO_outputs = output_copy
            run_global_vars.waitset = waitset_copy
            run_global_vars.signalset = signalset_copy
        elif lex == "/\\":
            assert r == 0
            if not lhs:
                return False
        if r == 0:
            rhs = self.rhs.eval(containers, boundedvars)
        else:
            lhs = self.lhs.eval(containers, boundedvars)

        # print("INFIX EVAL", lex, lhs, rhs)

        try:
            if lex == "/\\":
                return lhs and rhs
            if lex == "\\/":
                return lhs or rhs
            if lex == "=":
                return lhs == rhs
            if lex == "/=" or lex == "#":
                return lhs != rhs
            if lex == "\\in":
                return lhs in rhs
            if lex == "\\notin":
                return lhs not in rhs
        except Exception as e:
            print("Evaluating infix", str(self.op), "failed")
            print(e)
            print(traceback.format_exc())
            raise PlusPyError("critical failure") from e

        logger.error(f"Infix operator {self.op} not defined")
        raise ValueError("Infix operator not defined")


# Apply the given arguments in vals to func
def funceval(func, vals):
    assert func is not None

    # strings are special case of functions
    if isinstance(func, str):
        assert len(vals) == 1
        assert vals[0] >= 1
        assert vals[0] <= len(func)
        return func[vals[0] - 1]

    # Turn function into a dictionary
    if isinstance(func, tuple):
        assert len(vals) == 1
        assert vals[0] >= 1
        assert vals[0] <= len(func)
        kvs = {(i + 1): func[i] for i in range(len(func))}
    else:
        assert isinstance(func, FrozenDict)
        kvs = func.d

    # See if there's a match against the kvs
    if len(vals) == 1:
        k = vals[0]
    else:
        k = tuple(vals)
    v = kvs.get(k)
    if v is not None:
        return v

    logger.error(f"FUNCEVAL {func} {vals} {kvs} {k}")
    raise ValueError("Function evaluation failed")


# v is either a string, a tuple of values, or a FrozenDict.
# Return a uniform representation such that if two values should
# be equal they have the same representation.  Strings are the preferred
# representation, then tuples, then sets, then records, then nonces.
def simplify(v):
    if len(v) == 0:
        return ""
    if isinstance(v, str):
        return v

    # See if it's a record that can be converted into a tuple
    if isinstance(v, FrozenDict):
        kvs = v.d
        if set(kvs.keys()) == set(range(1, len(v) + 1)):
            t = []
            for i in range(1, len(v) + 1):
                t += [kvs[i]]
            v = tuple(t)

    # See if it's a tuple that can be converted into a string:
    if isinstance(v, tuple) and all(isinstance(c, str) and len(c) == 1 for c in v):
        return "".join(v)

    return v


class IndexExpression(Expression):
    def __init__(self, token=None, func=None, args=None, primed=None):
        self.token = token
        self.func = func
        self.args = args
        self.primed = primed

    def from_ast(self, expr):
        (token, func, args) = expr
        self.token = token
        self.func = compile_expression(func)
        self.primed = self.func.primed
        (t, a) = args
        assert t == "CommaList"
        self.args = []
        for ast in a:
            ca = compile_expression(ast)
            if ca.primed:
                self.primed = True
            self.args += [ca]
        assert self.args != []
        return self

    def __str__(self):
        assert self.args != []
        result = ""
        for x in self.args:
            if result != "":
                result += ", "
            result += x.__str__()
        return "Index(" + self.func.__str__() + ", [" + result + "])"

    def substitute(self, subs):
        assert self.args != []
        func = self.func.substitute(subs)
        args = [arg.substitute(subs) for arg in self.args]
        assert args != []
        return IndexExpression(func=func, args=args, primed=self.primed)

    def eval(self, containers, boundedvars):
        assert self.args != []
        args = [arg.eval(containers, boundedvars) for arg in self.args]
        r = self.func.apply(containers, boundedvars, args)
        assert r is not None
        return r


class TupleExpression(Expression):
    def __init__(self, exprs=None, primed=False):
        self.exprs = exprs
        self.primed = primed

    def from_ast(self, ast):
        self.primed = False
        (t, a) = ast
        assert t == "Maybe"
        if a is None:
            self.exprs = []
        else:
            (t1, a1) = a
            assert t1 == "CommaList"
            self.exprs = [compile_expression(x) for x in a1]
            for e in self.exprs:
                if e.primed:
                    self.primed = True
                    break
        return self

    def __str__(self):
        result = ""
        for x in self.exprs:
            if result != "":
                result += ", "
            if x is None:
                result += "None"
            else:
                result += x.__str__()
        return "Tuple(" + result + ")"

    def substitute(self, subs):
        return TupleExpression(exprs=[e.substitute(subs) for e in self.exprs], primed=self.primed)

    def eval(self, containers, boundedvars):
        return simplify(tuple([e.eval(containers, boundedvars) for e in self.exprs]))

    def apply(self, containers, boundedvars, fargs):
        assert len(fargs) == 1
        # print("ZZZZ", [x.eval(containers, boundedvars) for x in self.exprs], fargs[0])
        return self.exprs[fargs[0] - 1].eval(containers, boundedvars)


class SetExpression(Expression):
    def __init__(self, elements=None, primed=False):
        self.elements = elements
        self.primed = primed

    def from_ast(self, ast):
        (t, a) = ast
        assert t == "Maybe"
        self.primed = False
        self.elements = []
        if a is not None:
            (t0, a0) = a
            assert t0 == "CommaList"
            for x in a0:
                cx = compile_expression(x)
                if cx.primed:
                    self.primed = True
                self.elements += [cx]
        return self

    def __str__(self):
        result = ""
        for x in self.elements:
            if result != "":
                result += ", "
            if x is None:
                result += "None"
            else:
                result += x.__str__()
        return "Set(" + result + ")"

    def substitute(self, subs):
        return SetExpression(elements=[e.substitute(subs) for e in self.elements], primed=self.primed)

    def eval(self, containers, boundedvars):
        result = set()
        for x in self.elements:
            result.add(x.eval(containers, boundedvars))
        return frozenset(result)


class FilterExpression(Expression):
    def __init__(self, vars=None, elements=None, expr=None, primed=False):
        self.vars = vars
        self.elements = elements
        self.expr = expr
        self.primed = primed

    def from_ast(self, filter):
        (t0, a0) = filter[0]
        if t0 == "Identifier":
            self.vars = [BoundvarExpression(a0.lexeme)]
        else:
            assert t0 == "Tuple"
            (t1, a1) = a0
            assert t1 == "CommaList"
            self.vars = [BoundvarExpression(v) for (t, v) in a1]
        self.elements = compile_expression(filter[1])
        name_stack.append({bv.id: bv for bv in self.vars})
        self.expr = compile_expression(filter[2])
        name_stack.pop()
        return self

    def __str__(self):
        return "Filter(" + str(self.vars) + ", " + self.elements.__str__() + ", " + self.expr.__str__() + ")"

    def substitute(self, subs):
        return FilterExpression(
            vars=self.vars,
            elements=self.elements.substitute(subs),
            expr=self.expr.substitute(subs),
            primed=self.primed,
        )

    def eval(self, containers, boundedvars):
        result = set()
        assert len(self.vars) == 1
        elements = self.elements.eval(containers, boundedvars)
        result = []
        bvs = boundedvars.copy()
        assert len(self.vars) == 1  # TODO

        # Need to go through elements in defined order because
        # of pseudo-randomization
        elements = sorted(elements, key=lambda x: key(x))

        for x in elements:
            bvs[self.vars[0]] = ValueExpression(x)
            if self.expr.eval(containers, bvs):
                result.append(x)
        return frozenset(result)


class NumberExpression(Expression):
    def __init__(self, n):
        self.number = int(n)
        self.primed = False

    def __str__(self):
        return "Number(" + str(self.number) + ")"

    def substitute(self, subs):
        return self

    def eval(self, containers, boundedvars):
        return self.number


class StringExpression(Expression):
    def __init__(self, s):
        self.string = s
        self.primed = False

    def __str__(self):
        return 'String("' + self.string + '")'

    def substitute(self, subs):
        return self

    def eval(self, containers, boundedvars):
        return self.string

    def apply(self, containers, boundedvars, fargs):
        assert len(fargs) == 1
        return self.string[fargs[0] - 1]


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Compiler: AST pretty printer
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# For print_ast: AST nodes that have lists of nodes as arguments
LIST_NODES = [
    "Concat",
    "Index",
    "GModule",
    "AtLeast0",
    "AtLeast1",
    "CommaList",
    "GOperatorDefinition",
    "GQuantifierBound",
    "op",
    "lambda",
    "except",
    "if",
    "forall",
    "exists",
    "square",
]

# print_ast: AST nodes that have another AST node as argument
TAG_NODES = [
    "GUnit",
    "GTheorem",
    "GBasicExpression",
    "GExpression18",
    "GArgument",
    "GVariableDeclaration",
    "GConstantDeclaration",
    "Tuple",
    "parentheses",
    "set",
    "wf",
    "sf",
]


# Pretty printer for AST.  Every node in the AST is of the form (t, a),
# where 't' is the type and 'a' is what's in the node
def print_ast(x, indent):
    (t, a) = x
    if not t:
        print("ERROR: " + str(a))
        return
    print(indent + "(" + t + ",", end="")
    if t in LIST_NODES:
        print()
        print(indent + ".[")
        for y in a:
            print_ast(y, indent + "..")
        print(indent + ".]")
        print(indent + ")")
    elif t in TAG_NODES:
        print()
        print_ast(a, indent + "..")
        print(indent + ")")
    elif t.startswith("Infix"):
        (op, lhs, rhs) = a
        print(" " + op + ":")
        print_ast(lhs, indent + "..")
        print_ast(rhs, indent + "..")
        print(indent + ")")
    elif t.startswith("Prefix"):
        (op, expr) = a
        print(" " + op + ":")
        print_ast(expr, indent + "..")
        print(indent + ")")
    elif t.startswith("Postfix"):
        (expr, op) = a
        print(" " + op + ":")
        print_ast(expr, indent + "..")
        print(indent + ")")
    elif t == "Maybe":
        if a is None:
            print(" None)")
        else:
            print()
            print_ast(a, indent + "..")
            print(indent + ")")
    else:
        print(" '" + str(a) + "'", end="")
        print(")")
