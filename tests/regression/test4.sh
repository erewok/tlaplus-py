$PLUSPY_EXEC -S0 -c100 TestChannel > test4.out 2>test4.out2
if cmp -s test4.out tests/regression/test4.exp
then
    if cmp -s test4.out2 tests/regression/test4.exp2
    then
        rm -rf test4.out test4.out2
        exit 0
    fi
fi
exit 1
