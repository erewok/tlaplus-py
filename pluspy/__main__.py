import getopt
import logging
import os
import sys
import threading

from .pluspy import exit, PlusPy
from . import parser
from .runners import run


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


pluspypath = ".:./modules/lib:./modules/book:./modules/other"

def main():
    global pluspypath

    if os.environ.get("PLUSPYPATH") is not None:
        pluspypath = os.environ["PLUSPYPATH"]

    # Get options.  First set default values
    initOp = "Init"
    nextOps = set()
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
            initOp = a
        elif o in { "-n", "--next"  }:
            nextOps.add(a)
        elif o in { "-P", "--path" }:
            pluspypath = a
        elif o in { "-s" }:
            silent = True
        elif o in { "-S", "--seed" }:
            seed = int(a)
        else:
            assert False, "unhandled option"
    if len(args) != 1:
        usage()

    parser.verbose = verbose
    parser.silent = silent
    pp = PlusPy(args[0], seed=seed)
    mod = pp.mod
    if verbose:
        logger.info(f"Loaded module: {mod.name}")

    if verbose:
        logger.info("\n")
        logger.info("---------------")
        logger.info("Initialize state")
        logger.info("---------------")
    pp.init(initOp)
    if not silent:
        logger.info(f"Initial context: {format(pp.getall())}")

    if verbose:
        logger.info("\n")
        logger.info("---------------")
        logger.info(f"Run behavior for {parser.maxcount} steps")
        logger.info("---------------")

    if len(nextOps) != 0:
        threads = set()
        for next in nextOps:
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
