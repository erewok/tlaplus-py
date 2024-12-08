$PLUSPY_EXEC -S0 -c100 TestInnerFIFO2 > test6.out 2>test6.out2
if cmp -s test6.out tests/regression/test6.exp
then
    if cmp -s test6.out2 tests/regression/test6.exp2
    then
        rm -rf test6.out test6.out2
        exit 0
    fi
fi
exit 1
