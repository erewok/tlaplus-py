import getopt
import logging
import os
import sys
import threading

from . import parser
from .pluspy import exit, PlusPy
from .runners import run
from .utils import val_to_string

logger = logging.getLogger("pluspy")
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler(sys.stdout)]


def usage():
    print("Usage: pluspy [options] tla-module")
    print("  options: ")
    print("    -c cnt: max #times that Next should be evaluated")
    print("    -h: help")
    print("    -i operator: Init operator (default Init)")
    print("    -n operator: Next operator (default Next)")
    print("    -P path: module directory search path")
    print("    -s: silent")
    print("    -S seed: random seed")
    print("    -v: verbose output")
    exit(1)


DEFAULT_MODULE_PATH = ".:./modules/lib:./modules/book:./modules/other"

def main():
    pluspypath = os.environ.get("PLUSPYPATH", DEFAULT_MODULE_PATH)
    # Get options.  First set default values
    init_op = "Init"
    next_ops = set()
    seed = None
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                        "c:hi:n:P:sS:v",
                        ["help", "init=", "next=", "path=", "seed="])
    except getopt.GetoptError as err:
        logger.error(str(err))
        usage()
    verbose = False
    silent = False
    for o, a in opts:
        if o in { "-v" }:
            verbose = True
        elif o in { "-c" }:
            parser.maxcount = int(a)
        elif o in { "-h", "--help" }:
            usage()
        elif o in { "-i", "--init" }:
            init_op = a
        elif o in { "-n", "--next"  }:
            next_ops.add(a)
        elif o in { "-P", "--path" }:
            pluspypath = a
        elif o in { "-s" }:
            silent = True
        elif o in { "-S", "--seed" }:
            seed = int(a)
        else:
            raise ValueError("unhandled option")
    if len(args) != 1:
        usage()

    parser.verbose = verbose
    parser.silent = silent
    pp = PlusPy(args[0], seed=seed, module_path=pluspypath  )
    mod = pp.mod
    if verbose:
        logger.info(f"Loaded module: {mod.name}")

    if verbose:
        logger.info("\n")
        logger.info("---------------")
        logger.info("Initialize state")
        logger.info("---------------")
    pp.init(init_op)
    if not silent:
        logger.info(f"Initial context: {val_to_string(pp.getall())}")

    if verbose:
        logger.info("\n")
        logger.info("---------------")
        logger.info(f"Run behavior for {parser.maxcount} steps")
        logger.info("---------------")

    if len(next_ops) != 0:
        threads = set()
        for next in next_ops:
            t = threading.Thread(target=run, args=(pp, next), kwargs={"silent": silent, "verbose": verbose})
            threads.add(t)
            t.start()
        for t in threads:
            t.join()
    else:
        run(pp, "Next", silent=silent, verbose=verbose)
    if not silent:
        logger.info("MAIN DONE")
    exit(0)

if __name__ == "__main__":
    main()
