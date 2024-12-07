import sys

from . import parser, wrappers
from .utils import convert, isnumeral, val_to_string
from .wrappers import netSender


def flush():
    for x in wrappers.IO_outputs:
        d = x.d
        if d["intf"] == "fd":
            if d["mux"] == "stdout":
                print(d["data"], end="")
                sys.stdout.flush()
            else:
                assert d["mux"] == "stderr"
                print(d["data"], end="", file=sys.stderr)
                sys.stderr.flush()
        elif d["intf"] == "tcp":
            netSender(d["mux"], d["data"]).start()
        else:
            assert d["intf"] == "local"
            wrappers.IO_inputs.append(x)

    wakeup = False
    for x in wrappers.signalset:
        if x in wrappers.waitset:
            wrappers.waitset.remove(x)
            wakeup = True
    if wakeup:
        parser.cond.notifyAll()

def drain():
    wrappers.IO_outputs = []
    wrappers.signalset = set()


def handleOutput(output):
    d = convert(output)
    if d["intf"] == "fd":
        if d["mux"] == "stdout":
            print(d["data"], end="")
        else:
            assert d["mux"] == "stderr"
            print(d["data"], end="", file=sys.stderr)
    else:
        print("GOT OUTPUT", d)
        assert False

# The Next operator, possibly with arguments separated by "%"
def run(pp, next, silent: bool = False, verbose: bool = False):
    args = next.split("%")
    assert 1 <= len(args) and len(args) <= 2
    if len(args) == 1:
        arg = ""
    elif all(isnumeral(c) for c in args[1]):
        arg = int(args[1])
    else:
        arg = args[1]
    while True:
        with parser.lock:
            tries = 0
            flush()     # do all the outputs
            drain()     # remove all outputs
            while not pp.next(args[0], arg):
                tries += 1
                if verbose:
                    print("TRY AGAIN", tries, flush=True)
                if tries > 100:
                    parser.cond.wait(0.2)
                drain()
                if parser.maxcount is not None and parser.step >= parser.maxcount:
                    break

            if parser.maxcount is not None and parser.step >= parser.maxcount:
                break
            if pp.unchanged():
                if not silent:
                    print("No state change after successful step", flush=True)
                break
            tries = 0
            if not silent:
                print("Next state:", parser.step, val_to_string(pp.getall()), flush=True)
            parser.step += 1

            # To implement JWait/JSignalReturn
            while arg in wrappers.waitset:
                parser.cond.wait(0.2)
