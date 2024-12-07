# just manual: https://github.com/casey/just#readme
_default:
    just --list

# Install dependencies used by this project
bootstrap default="3.12":
    uv venv --python {{default}}
    just sync

# Sync dependencies with environment
sync:
    uv sync

# Build the project as a package (uv build)
build *args:
    uv build

# Run the code formatter
format:
    uv run ruff format pluspy tests

# Run code quality checks
check:
    #!/bin/bash -eux
    uv run ruff check pluspy tests

# Run mypy checks
check-types:
    #!/bin/bash -eux
    uv run mypy pluspy

# Run all tests locally
test *args:
    #!/bin/bash -eux
    uv run pytest {{args}}

# Run the project tests for CI environment (e.g. with code coverage)
ci-test coverage_dir='./coverage':
    uv run pytest --cov=pluspy --cov-report xml --junitxml=./coverage/unittest.junit.xml

# Run a spec
run *args:
    #!/bin/bash -eux
    ./pluspy.sh {{args}}

# Run the regression tests
regressions:
    #!/bin/bash -Eeux
    export PLUSPY_EXEC=./pluspy.sh

    for i in 1 2 3 4 5 6 7 9 10 11
    do
        echo running test $i
        ./tests/regression/test$i.sh
        case $? in
        0)
            ;;
        *)
            echo test $i failed
            exit 1
        esac
    done
