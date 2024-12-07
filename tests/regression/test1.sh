$PLUSPY_EXEC -S0 -c100 HourClock > test1.out 2>test1.out2
if cmp -s test1.out tests/regression/test1.exp
then
    if cmp -s test1.out2 tests/regression/test1.exp2
    then
        rm -rf test1.out test1.out2
        echo 'test1 passed'
        exit 0
    fi
fi
echo 'test1 failed'
exit 1
