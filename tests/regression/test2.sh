$PLUSPY_EXEC -S0 -c100 Peterson > test2.out 2>test2.out2
if cmp -s test2.out tests/regression/test2.exp
then
    if cmp -s test2.out2 tests/regression/test2.exp2
    then
        rm -rf test2.out test2.out2
        exit 0
    fi
fi
exit 1
