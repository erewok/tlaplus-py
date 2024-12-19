import argparse
import logging
import os
import sys
import threading

from . import run_global_vars
from .pluspy import PlusPy
from .runners import run
from .utils import val_to_string

logger = logging.getLogger("pluspy")
logger.setLevel(logging.INFO)


DEFAULT_MODULE_PATH = ".:./modules/lib:./modules/book:./modules/other"


def cli():
    """Parse command line arguments."""
    pluspypath = os.environ.get("PLUSPYPATH", DEFAULT_MODULE_PATH)
    parser = argparse.ArgumentParser(description="Run a TLA+ module")
    parser.add_argument("module", help="The TLA+ module to run")
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=1,
        help="The maximum number of times to evaluate Next",
    )
    parser.add_argument(
        "-i",
        "--init",
        default="Init",
        help="The name of the Init operator to run",
    )
    parser.add_argument(
        "-n",
        "--next",
        action="append",
        default="Next",
        help="The name of the Next operator to run",
    )
    parser.add_argument(
        "-P",
        "--path",
        default=pluspypath,
        help="The path to search for TLA+ modules",
    )
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        help="The random seed to use",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Suppress output",
    )
    return parser


def run_with_args(args):
    """Primary runner: separated here for easier regression testing"""
    run_global_vars.maxcount = args.count
    run_module = args.module
    pluspypath = args.path
    # Get options.  First set default values
    init_op = args.init
    next_ops = set()
    if args.next:
        next_ops.add(args.next)

    verbose = args.verbose
    silent = args.silent

    pp = PlusPy(run_module, pluspypath, seed=args.seed, verbose=verbose, silent=silent)
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
        logger.info(f"Run behavior for {run_global_vars.maxcount} steps")
        logger.info("---------------")

    if next_ops:
        threads = set()
        for next in next_ops:
            t = threading.Thread(
                target=run,
                args=(pp, next),
                kwargs={"silent": silent, "verbose": verbose},
            )
            threads.add(t)
            t.start()
        for t in threads:
            t.join()
    else:
        run(pp, "Next", silent=silent, verbose=verbose)
    if not silent:
        logger.info("MAIN DONE")


def main():
    """Main entry point for the pluspy command line tool."""
    logger.handlers = [logging.StreamHandler(sys.stderr)]
    parser = cli()
    args = parser.parse_args()
    return run_with_args(args)
