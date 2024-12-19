import logging
import sys

from . import run_global_vars
from .utils import convert, isnumeral, val_to_string
from .wrappers import NetSender

logger = logging.getLogger(__name__)


def flush():
    for x in run_global_vars.IO_outputs:
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
            NetSender(d["mux"], d["data"]).start()
        else:
            assert d["intf"] == "local"
            run_global_vars.IO_inputs.append(x)

    wakeup = False
    for x in run_global_vars.signalset:
        if x in run_global_vars.waitset:
            run_global_vars.waitset.remove(x)
            wakeup = True
    if wakeup:
        run_global_vars.cond.notifyAll()


def drain():
    run_global_vars.IO_outputs = []
    run_global_vars.signalset = set()


def handle_output(output):
    d = convert(output)
    if d["intf"] == "fd":
        if d["mux"] == "stdout":
            logger.info(d["data"], end="")
        else:
            assert d["mux"] == "stderr"
            logger.info(d["data"], end="", file=sys.stderr)
    else:
        logger.info("GOT OUTPUT", d)
        raise ValueError("Unknown output interface")


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

    step = 0
    while True:
        with run_global_vars.lock:
            tries = 0
            flush()  # do all the outputs
            drain()  # remove all outputs
            while not pp.next(args[0], arg) and run_global_vars.checkcontinue(step):
                tries += 1
                if verbose:
                    logger.info(f"TRY AGAIN {tries}")
                if tries > 100:
                    run_global_vars.cond.wait(0.2)
                drain()

            if run_global_vars.checkstop(step):
                break

            if pp.unchanged():
                if not silent:
                    logger.info("No state change after successful step")
                break
            tries = 0
            if not silent:
                logger.info(f"Next state: {step} {val_to_string(pp.getall())}")
            step += 1

            # To implement JWait/JSignalReturn
            while arg in run_global_vars.waitset:
                run_global_vars.cond.wait(0.2)
