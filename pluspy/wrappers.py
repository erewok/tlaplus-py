import pickle
import random
import socket
import threading
import time

from .utils import convert, simplify, FrozenDict

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
####    Python Wrappers (to replace TLA+ operator definitions)
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

# This is a dictionary of wrappers around Python functions
# Maps module names to dictionaries of (operator name, Wrapper) pairs
wrappers = {}


class Wrapper:
    def eval(self, id, args):
        assert False


class InfixWrapper(Wrapper):
    def __str__(self):
        return "Naturals!InfixWrapper()"

    def eval(self, id, args):
        assert len(args) == 2
        lhs = args[0]
        rhs = args[1]
        if id[0] == "\\":
            if id == "\\/":
                return lhs or rhs
            if id == "\\equiv":
                return lhs == rhs
            if id == "\\geq":
                return lhs >= rhs
            if id == "\\in":
                return lhs in rhs
            if id == "\\notin":
                return lhs not in rhs
            if id == "\\leq":
                return lhs <= rhs
            if id == "\\subset":
                return lhs.issubset(rhs) and lhs != rhs
            if id == "\\subseteq":
                return lhs.issubset(rhs)
            if id == "\\supset":
                return rhs.issubset(lhs) and rhs != lhs
            if id == "\\supseteq":
                return rhs.issubset(lhs)
            if id == "\\":
                return lhs.difference(rhs)
            if id in {"\\cap", "\\intersect"}:
                return lhs.intersection(rhs)
            if id in {"\\cup", "\\union"}:
                return lhs.union(rhs)
            if id == "\\div":
                return lhs // rhs
        else:
            if id == "/\\":
                return lhs and rhs
            if id == "=>":
                return (not lhs) or rhs
            if id == "<=>":
                return lhs == rhs
            if id in {"#", "/="}:
                return lhs != rhs
            if id == "<":
                return lhs < rhs
            if id == "=":
                return lhs == rhs
            if id == ">":
                return lhs > rhs
            if id == ">=":
                return lhs >= rhs
            if id in {"<=", "=<"}:
                return lhs <= rhs
            if id == "..":
                return frozenset({i for i in range(lhs, rhs + 1)})
            if id == "+":
                return lhs + rhs
            if id == "-":
                return lhs - rhs
            if id == "*":
                return lhs * rhs
            if id == "/":
                return lhs / rhs
            if id == "%":
                return lhs % rhs
            if id == "^":
                return lhs**rhs
        assert False


class OutfixWrapper(Wrapper):
    def __str__(self):
        return "Naturals!OutfixWrapper()"

    def subset_enum(self, lst, record, result):
        if lst == []:
            result.add(frozenset(record))
        else:
            self.subset_enum(lst[1:], record, result)
            self.subset_enum(lst[1:], record.union({lst[0]}), result)

    def eval(self, id, args):
        assert len(args) == 1
        expr = args[0]

        if id == "DOMAIN":
            if isinstance(expr, str) or isinstance(expr, tuple):
                return frozenset(range(1, len(expr) + 1))
            else:
                assert isinstance(expr, FrozenDict)
                return frozenset(expr.d.keys())

        if id == "UNION":
            result = set()
            for x in expr:
                result = result.union(x)
            return frozenset(result)

        if id == "SUBSET":
            result = set()
            self.subset_enum(list(expr), set(), result)
            return frozenset(result)

        # if id == "-.": return -expr
        if id in {"~", "\\lnot", "\\neg"}:
            return not expr

        assert False


wrappers["Core"] = {
    "=>": InfixWrapper(),
    "<=>": InfixWrapper(),
    "\\equiv": InfixWrapper(),
    # "/\\": InfixWrapper(),
    # "\\/": InfixWrapper(),
    "#": InfixWrapper(),
    "/=": InfixWrapper(),
    # "=": InfixWrapper(),
    # "\\in": InfixWrapper(),
    # "\\notin": InfixWrapper(),
    "\\subset": InfixWrapper(),
    "\\subseteq": InfixWrapper(),
    "\\supset": InfixWrapper(),
    "\\supseteq": InfixWrapper(),
    "\\": InfixWrapper(),
    "\\cap": InfixWrapper(),
    "\\intersect": InfixWrapper(),
    "\\cup": InfixWrapper(),
    "\\union": InfixWrapper(),
    "DOMAIN": OutfixWrapper(),
    "~": OutfixWrapper(),
    "\\lnot": OutfixWrapper(),
    "\\neg": OutfixWrapper(),
    "UNION": OutfixWrapper(),
    "SUBSET": OutfixWrapper(),
}

wrappers["Naturals"] = {
    "<": InfixWrapper(),
    ">": InfixWrapper(),
    ">=": InfixWrapper(),
    "\\geq": InfixWrapper(),
    "<=": InfixWrapper(),
    "=<": InfixWrapper(),
    "\\leq": InfixWrapper(),
    "..": InfixWrapper(),
    "+": InfixWrapper(),
    "-": InfixWrapper(),
    "*": InfixWrapper(),
    "/": InfixWrapper(),
    "\\div": InfixWrapper(),
    "%": InfixWrapper(),
    "^": InfixWrapper(),
}


class LenWrapper(Wrapper):
    def __str__(self):
        return "Sequences!Len(_)"

    def eval(self, id, args):
        assert len(args) == 1
        assert isinstance(args[0], tuple) or isinstance(args[0], str)
        return len(args[0])


class ConcatWrapper(Wrapper):
    def __str__(self):
        return "Sequences!Concat(_)"

    def eval(self, id, args):
        assert len(args) == 2
        return simplify(tuple(list(args[0]) + list(args[1])))


class AppendWrapper(Wrapper):
    def __str__(self):
        return "Sequences!Append(_)"

    def eval(self, id, args):
        assert len(args) == 2
        return simplify(tuple(list(args[0]) + [args[1]]))


wrappers["Sequences"] = {
    "Len": LenWrapper(),
    "\\o": ConcatWrapper(),
    "Append": AppendWrapper(),
}


class AssertWrapper(Wrapper):
    def __str__(self):
        return "TLC!Assert(_, _)"

    def eval(self, id, args):
        assert len(args) == 2
        assert args[0], args[1]
        return True


class JavaTimeWrapper(Wrapper):
    def __str__(self):
        return "TLC!JavaTime()"

    def eval(self, id, args):
        assert len(args) == 0
        return int(time.time() * 1000)


class PrintWrapper(Wrapper):
    def __str__(self):
        return "TLC!Print(_)"

    def eval(self, id, args):
        assert len(args) == 2
        print(str(convert(args[0])), end="")
        return args[1]


class PrintTWrapper(Wrapper):
    def __str__(self):
        return "TLC!TPrint(_)"

    def eval(self, id, args):
        assert len(args) == 1
        print(str(convert(args[0])))
        return True


class RandomElementWrapper(Wrapper):
    def __str__(self):
        return "TLC!RandomElement(_)"

    def eval(self, id, args):
        assert len(args) == 1
        lst = list(args[0])
        r = random.randrange(len(lst))
        return lst[r]


class ToStringWrapper(Wrapper):
    def __str__(self):
        return "TLC!ToString(_)"

    def eval(self, id, args):
        assert len(args) == 1
        return str(format(args[0]))


TLCvars = {}


class TLCSetWrapper(Wrapper):
    def __str__(self):
        return "TLC!TLCSet(_, _)"

    def eval(self, id, args):
        assert len(args) == 2
        TLCvars[args[0]] = args[1]
        return True


class TLCGetWrapper(Wrapper):
    def __str__(self):
        return "TLC!TLCGet(_)"

    def eval(self, id, args):
        assert len(args) == 1
        return TLCvars[args[0]]


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


class JWaitWrapper(Wrapper):
    def __str__(self):
        return "TLC!JWait(_)"

    def eval(self, id, args):
        assert len(args) == 1
        global waitset
        assert args[0] not in waitset
        waitset.add(args[0])
        return True


class JSignalReturnWrapper(Wrapper):
    def __str__(self):
        return "TLC!JSignalReturn(_,_)"

    def eval(self, id, args):
        assert len(args) == 2
        global signalset
        signalset.add(args[0])
        return args[1]


wrappers["TLCExt"] = {"JWait": JWaitWrapper(), "JSignalReturn": JSignalReturnWrapper()}


class netReceiver(threading.Thread):
    def __init__(self, src, mux, verbose: bool=False):
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
                print("netReceiver", addr, msg)
            IO_inputs.append(FrozenDict({"intf": "tcp", "mux": self.mux, "data": msg}))
            cond.notify()


class netSender(threading.Thread):
    def __init__(self, mux, msg):
        threading.Thread.__init__(self)
        self.mux = mux
        self.msg = msg

    def run(self, verbose=False):
        parts = self.mux.split(":")
        dst = (parts[0], int(parts[1]))
        if verbose:
            print("netSender", dst, self.msg)
        while True:
            try:
                skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                skt.connect(dst)
                skt.sendall(pickle.dumps(self.msg))
                skt.close()
                break
            except ConnectionRefusedError:
                time.sleep(0.5)


class netServer(threading.Thread):
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
            netReceiver(client, self.mux).start()


IO_inputs = []
IO_outputs = []
IO_running = set()


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

    def eval(self, id, args):
        assert len(args) == 3
        IO_outputs.append(
            FrozenDict({"intf": args[0], "mux": args[1], "data": args[2]})
        )
        return True


class IOWaitWrapper(Wrapper):
    def __str__(self):
        return "IOUtils!IOWait(Pattern(_))"

    def eval(self, id, args):
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
                netServer(args[1]).start()
            else:
                assert args[0] == "local"
            IO_running.add((args[0], args[1]))

        return False


class IOGetWrapper(Wrapper):
    def __str__(self):
        return "IOUtils!IOGet(Pattern(_))"

    def eval(self, id, args):
        assert len(args) == 2
        for x in IO_inputs:
            assert isinstance(x, FrozenDict)
            d = x.d
            if d["intf"] == args[0] and d["mux"] == args[1]:
                IO_inputs.remove(x)
                return d["data"]
        assert False


wrappers["IOUtils"] = {
    "IOPut": IOPutWrapper(),
    "IOWait": IOWaitWrapper(),
    "IOGet": IOGetWrapper(),
}
