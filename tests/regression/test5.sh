$PLUSPY_EXEC -S0 -c100 TestInnerFIFO > test5.out 2>test5.out2
if cmp -s test5.out tests/regression/test5.exp
then
    if cmp -s test5.out2 tests/regression/test5.exp2
    then
        rm -rf test5.out test5.out2
        exit 0
    fi
fi
exit 1
