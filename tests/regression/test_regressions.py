import logging
import os
from io import StringIO

import pytest

from pluspy import cli

directory = os.path.dirname(__file__)

DEFAULT_SEED = "-S0"
DEFAULT_COUNT = "-c100"
CUSTOM_INIT_NAME = "Spec"

REGRESSION_TEST_WITH_CUSTOM_INIT = [
    "TestFIFO",
    "TestBoundedFIFO",
]

REGRESSION_TESTS = [
    "HourClock",
    # Fails due to WF_vars in spec
    "Peterson",
    "Prime",
    "TestChannel",
    "TestInnerFIFO",
    "TestInnerFIFO2",
    "Qsort",
    "Exprs",
    # Hangs pytest at finish due to seeming threading deadlock
    # "TestBinBosco",
]


def expected_output_loader(name: str) -> str:
    expect_file = f"{name}.exp"
    with open(os.path.join(directory, expect_file)) as fl:
        return fl.read()


@pytest.fixture()
def log_capture():
    log_capture_fl = StringIO()
    handler = logging.StreamHandler(log_capture_fl)
    # Attach to root logger
    cli.logger.addHandler(handler)
    yield log_capture_fl


@pytest.mark.parametrize("test_name", REGRESSION_TESTS)
def test_regression_common(test_name, log_capture):
    parser = cli.cli()
    args = parser.parse_args([test_name, DEFAULT_SEED, DEFAULT_COUNT])
    breakpoint()
    cli.run_with_args(args)
    log_contents = log_capture.getvalue()
    expected = expected_output_loader(test_name)
    assert log_contents == expected


@pytest.mark.parametrize("test_name", REGRESSION_TEST_WITH_CUSTOM_INIT)
def test_regression_custom_init(test_name, log_capture):
    logout = StringIO()
    logging.basicConfig(level=logging.INFO, stream=logout)
    parser = cli.cli()
    args = parser.parse_args([test_name, DEFAULT_SEED, DEFAULT_COUNT, "-i", "Spec"])
    cli.run_with_args(args)
    log_contents = log_capture.getvalue()
    expected = expected_output_loader(test_name)
    assert log_contents == expected
