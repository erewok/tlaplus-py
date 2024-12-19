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
    #!/bin/bash -u
    export PLUSPY_EXEC=./pluspy.sh
    failing="2 7 8 9 10 11"
    for i in 1 3 4 5 6
    do
        echo running test $i
        ./tests/regression/test$i.sh
        case $? in
        0)
            echo test $i passed
            ;;
        *)
            echo test $i failed
            echo "To run the failing test -> PLUSPY_EXEC=./pluspy.sh ./tests/regression/test$i.sh"
            exit 1
        esac
    done
