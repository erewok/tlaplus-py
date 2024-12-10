import pickle
import random
import socket
import threading
import time
from abc import ABC, abstractmethod

from .lexer import InfixTokenKind
from .utils import convert, FrozenDict, simplify, val_to_string

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Python Wrappers (to replace TLA+ operator definitions)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# This is a dictionary of wrappers around Python functions
# Maps module names to dictionaries of (operator name, Wrapper) pairs
signalset = set()
waitset = set()
TLCvars = {}
IO_inputs = []
IO_outputs = []
IO_running = set()


class Wrapper(ABC):
    @abstractmethod
    def __call__(self, ident, args):
        raise NotImplementedError("eval not implemeted for base class")


class InfixWrapper(Wrapper):
    def __str__(self):
        return "Core!InfixWrapper()"

    def __call__(self, ident, args):
        """Evaluate an infix operator into a Python value."""
        assert len(args) == 2, "Infix must have left-hand and right-hand args"
        lhs = args[0]
        rhs = args[1]
        match ident:
            case InfixTokenKind.And.lexeme:
                return lhs or rhs
            case InfixTokenKind.Equivalent.lexeme:
                return lhs == rhs
            case InfixTokenKind.GreaterEqual.lexeme:
                return lhs >= rhs
            case InfixTokenKind.In.lexeme:
                return lhs in rhs
            case InfixTokenKind.NotIn.lexeme:
                return lhs not in rhs
            case InfixTokenKind.LessEqual.lexeme:
                return lhs <= rhs
            case InfixTokenKind.Subset.lexeme:
                return lhs.issubset(rhs) and lhs != rhs
            case InfixTokenKind.SubsetEqual.lexeme:
                return lhs.issubset(rhs)
            case InfixTokenKind.Superset.lexeme:
                return rhs.issubset(lhs) and rhs != lhs
            case InfixTokenKind.SupersetEqual.lexeme:
                return rhs.issubset(lhs)
            case InfixTokenKind.Backslash.lexeme:
                return lhs.difference(rhs)
            case InfixTokenKind.Cap.lexeme | InfixTokenKind.Intersect.lexeme:
                return lhs.intersection(rhs)
            case InfixTokenKind.Cup.lexeme | InfixTokenKind.Union.lexeme:
                return lhs.union(rhs)
            case InfixTokenKind.Divide.lexeme:
                return lhs // rhs
            case InfixTokenKind.And.lexeme:
                return lhs and rhs
            case InfixTokenKind.RightImplies.lexeme:
                return (not lhs) or rhs
            case InfixTokenKind.LeftImplies.lexeme:
                return (not rhs) or lhs
            case InfixTokenKind.MutualImplies.lexeme:
                # The value of the expression A <=> B is defined for A \in BOOLEAN /\ B \in BOOLEAN.
                # For non-Boolean values of A and B, the meaning of the operator <=> is unspecified by TLA+.
                return (lhs == rhs) and (lhs in {True, False})
            case InfixTokenKind.Hash.lexeme | InfixTokenKind.NotEqual.lexeme:
                return lhs != rhs
            case InfixTokenKind.LessThan.lexeme:
                return lhs < rhs
            case InfixTokenKind.Equal.lexeme:
                return lhs == rhs
            case InfixTokenKind.GreaterThan.lexeme:
                return lhs > rhs
            case InfixTokenKind.GreaterThanEqual.lexeme:
                return lhs >= rhs
            case InfixTokenKind.LessThanEqual.lexeme:
                return lhs <= rhs
            case InfixTokenKind.DotDot.lexeme:
                return frozenset({i for i in range(lhs, rhs + 1)})
            case InfixTokenKind.Plus.lexeme:
                return lhs + rhs
            case InfixTokenKind.Minus.lexeme:
                return lhs - rhs
            case InfixTokenKind.Star.lexeme:
                return lhs * rhs
            case InfixTokenKind.Slash.lexeme:
                return lhs / rhs
            case InfixTokenKind.Percent.lexeme:
                return lhs % rhs
            case InfixTokenKind.Caret.lexeme:
                return lhs**rhs
            case _:
                raise ValueError(f"Unknown operator {ident}")


class OutfixWrapper(Wrapper):
    def __str__(self):
        return "Naturals!OutfixWrapper()"

    def subset_enum(self, lst, record, result):
        if lst == []:
            result.add(frozenset(record))
        else:
            self.subset_enum(lst[1:], record, result)
            self.subset_enum(lst[1:], record.union({lst[0]}), result)

    def __call__(self, ident, args):
        assert len(args) == 1
        expr = args[0]

        if ident == "DOMAIN":
            if isinstance(expr, str) or isinstance(expr, tuple):
                return frozenset(range(1, len(expr) + 1))
            else:
                assert isinstance(expr, FrozenDict)
                return frozenset(expr.d.keys())

        if ident == "UNION":
            result = set()
            for x in expr:
                result = result.union(x)
            return frozenset(result)

        if ident == "SUBSET":
            result = set()
            self.subset_enum(list(expr), set(), result)
            return frozenset(result)

        # if ident == "-.": return -expr
        if ident in {"~", "\\lnot", "\\neg"}:
            return not expr

        raise ValueError(f"Unknown operator {id}")


class LenWrapper(Wrapper):
    def __str__(self):
        return "Sequences!Len(_)"

    def __call__(self, ident, args):
        assert len(args) == 1
        assert isinstance(args[0], tuple) or isinstance(args[0], str)
        return len(args[0])


class ConcatWrapper(Wrapper):
    def __str__(self):
        return "Sequences!Concat(_)"

    def __call__(self, ident, args):
        assert len(args) == 2
        return simplify(tuple(list(args[0]) + list(args[1])))


class AppendWrapper(Wrapper):
    def __str__(self):
        return "Sequences!Append(_)"

    def __call__(self, _ident, args):
        assert len(args) == 2
        return simplify(tuple(list(args[0]) + [args[1]]))


class AssertWrapper(Wrapper):
    def __str__(self):
        return "TLC!Assert(_, _)"

    def __call__(self, ident, args):
        assert len(args) == 2
        assert args[0], args[1]
        return True


class JavaTimeWrapper(Wrapper):
    def __str__(self):
        return "TLC!JavaTime()"

    def __call__(self, ident, args):
        assert len(args) == 0
        return int(time.time() * 1000)


class PrintWrapper(Wrapper):
    def __str__(self):
        return "TLC!Print(_)"

    def __call__(self, ident, args):
        assert len(args) == 2
        print(str(convert(args[0])), end="")
        return args[1]


class PrintTWrapper(Wrapper):
    def __str__(self):
        return "TLC!TPrint(_)"

    def __call__(self, ident, args):
        assert len(args) == 1
        print(str(convert(args[0])))
        return True


class RandomElementWrapper(Wrapper):
    def __str__(self):
        return "TLC!RandomElement(_)"

    def __call__(self, ident, args):
        assert len(args) == 1
        lst = list(args[0])
        r = random.randrange(len(lst))
        return lst[r]


class ToStringWrapper(Wrapper):
    def __str__(self):
        return "TLC!ToString(_)"

    def __call__(self, ident, args):
        assert len(args) == 1
        return val_to_string(args[0])


class TLCSetWrapper(Wrapper):
    def __str__(self):
        return "TLC!TLCSet(_, _)"

    def __call__(self, ident, args):
        assert len(args) == 2
        TLCvars[args[0]] = args[1]
        return True


class TLCGetWrapper(Wrapper):
    def __str__(self):
        return "TLC!TLCGet(_)"

    def __call__(self, ident, args):
        assert len(args) == 1
        return TLCvars[args[0]]


class JWaitWrapper(Wrapper):
    def __str__(self):
        return "TLC!JWait(_)"

    def __call__(self, ident, args):
        assert len(args) == 1
        global waitset
        assert args[0] not in waitset
        waitset.add(args[0])
        return True


class JSignalReturnWrapper(Wrapper):
    def __str__(self):
        return "TLC!JSignalReturn(_,_)"

    def __call__(self, ident, args):
        assert len(args) == 2
        global signalset
        signalset.add(args[0])
        return args[1]


class NetReceiver(threading.Thread):
    def __init__(self, src, mux, verbose: bool = False):
        threading.Thread.__init__(self)
        self.src = src
        self.mux = mux
        self.verbose = verbose

    def run(self, lock, cond, verbose=False):
        global IO_inputs

        (skt, addr) = self.src
        (host, port) = addr
        all = []
        while True:
            data = skt.recv(8192)
            if not data:
                break
            all.append(data)
        with lock:
            msg = pickle.loads(b"".join(all))
            if verbose:
                print("NetReceiver", addr, msg)
            IO_inputs.append(FrozenDict({"intf": "tcp", "mux": self.mux, "data": msg}))
            cond.notify()


class NetSender(threading.Thread):
    def __init__(self, mux, msg):
        threading.Thread.__init__(self)
        self.mux = mux
        self.msg = msg

    def run(self, verbose=False):
        parts = self.mux.split(":")
        dst = (parts[0], int(parts[1]))
        if verbose:
            print("NetSender", dst, self.msg)
        while True:
            try:
                skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                skt.connect(dst)
                skt.sendall(pickle.dumps(self.msg))
                skt.close()
                break
            except ConnectionRefusedError:
                time.sleep(0.5)


class NetServer(threading.Thread):
    def __init__(self, mux):
        threading.Thread.__init__(self)
        self.mux = mux

    def run(self):
        skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        parts = self.mux.split(":")
        skt.bind((parts[0], int(parts[1])))
        skt.listen()
        while True:
            client = skt.accept()
            NetReceiver(client, self.mux).start()


class Reader(threading.Thread):
    def run(self, lock, cond):
        global IO_inputs

        while True:
            inp = input()
            with lock:
                IO_inputs.append(
                    FrozenDict({"intf": "fd", "mux": "stdin", "data": inp})
                )
                cond.notify()


class IOPutWrapper(Wrapper):
    def __str__(self):
        return "IOUtils!IOPut(_)"

    def __call__(self, ident, args):
        assert len(args) == 3
        IO_outputs.append(
            FrozenDict({"intf": args[0], "mux": args[1], "data": args[2]})
        )
        return True


class IOWaitWrapper(Wrapper):
    def __str__(self):
        return "IOUtils!IOWait(Pattern(_))"

    def __call__(self, ident, args):
        global IO_running

        assert len(args) == 2

        # First check if there's already input
        for x in IO_inputs:
            assert isinstance(x, FrozenDict)
            d = x.d
            if d["intf"] == args[0] and d["mux"] == args[1]:
                return True

        # If not, make sure a reader/receiver is running
        if (args[0], args[1]) not in IO_running:
            if args[0] == "fd" and args[1] == "stdin":
                Reader().start()
            elif args[0] == "tcp":
                NetServer(args[1]).start()
            else:
                assert args[0] == "local"
            IO_running.add((args[0], args[1]))

        return False


class IOGetWrapper(Wrapper):
    def __str__(self):
        return "IOUtils!IOGet(Pattern(_))"

    def __call__(self, ident, args):
        assert len(args) == 2
        for x in IO_inputs:
            assert isinstance(x, FrozenDict)
            d = x.d
            if d["intf"] == args[0] and d["mux"] == args[1]:
                IO_inputs.remove(x)
                return d["data"]
        raise ValueError(f"IOGet.eval({args[0]}, {args[1]}) failed")


def build_wrappers() -> dict:
    # Eventually we can turn these into functions
    # For now, singletons are fine, instead of what the code
    # did previously: create a new instance of each wrapper for every operator.
    infix_inst = InfixWrapper()
    outfix_inst = OutfixWrapper()
    len_inst = LenWrapper()
    concat_inst = ConcatWrapper()
    append_inst = AppendWrapper()

    wrappers = {}
    wrappers["Core"] = {
        "=>": infix_inst,
        "<=>": infix_inst,
        "\\equiv": infix_inst,
        # "/\\": infix_inst,
        # "\\/": infix_inst,
        "#": infix_inst,
        "/=": infix_inst,
        # "=": infix_inst,
        # "\\in": infix_inst,
        # "\\notin": infix_inst,
        "\\subset": infix_inst,
        "\\subseteq": infix_inst,
        "\\supset": infix_inst,
        "\\supseteq": infix_inst,
        "\\": infix_inst,
        "\\cap": infix_inst,
        "\\intersect": infix_inst,
        "\\cup": infix_inst,
        "\\union": infix_inst,
        "DOMAIN": outfix_inst,
        "~": outfix_inst,
        "\\lnot": outfix_inst,
        "\\neg": outfix_inst,
        "UNION": outfix_inst,
        "SUBSET": outfix_inst,
    }

    wrappers["Naturals"] = {
        "<": infix_inst,
        ">": infix_inst,
        ">=": infix_inst,
        "\\geq": infix_inst,
        "<=": infix_inst,
        "=<": infix_inst,
        "\\leq": infix_inst,
        "..": infix_inst,
        "+": infix_inst,
        "-": infix_inst,
        "*": infix_inst,
        "/": infix_inst,
        "\\div": infix_inst,
        "%": infix_inst,
        "^": infix_inst,
    }

    wrappers["Sequences"] = {
        "Len": len_inst,
        "\\o": concat_inst,
        "Append": append_inst,
    }
    # These are only instantiated once, so we can use the same instance
    wrappers["TLC"] = {
        "Assert": AssertWrapper(),
        "JavaTime": JavaTimeWrapper(),
        "Print": PrintWrapper(),
        "PrintT": PrintTWrapper(),
        "RandomElement": RandomElementWrapper(),
        "TLCSet": TLCSetWrapper(),
        "TLCGet": TLCGetWrapper(),
        "ToString": ToStringWrapper(),
    }
    wrappers["TLCExt"] = {
        "JWait": JWaitWrapper(),
        "JSignalReturn": JSignalReturnWrapper(),
    }

    wrappers["IOUtils"] = {
        "IOPut": IOPutWrapper(),
        "IOWait": IOWaitWrapper(),
        "IOGet": IOGetWrapper(),
    }
    return wrappers
