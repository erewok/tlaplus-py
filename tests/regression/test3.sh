$PLUSPY_EXEC  -S0 -c100 Prime > test3.out 2>test3.out2
if cmp -s test3.out tests/regression/test3.exp
then
    if cmp -s test3.out2 tests/regression/test3.exp2
    then
        rm -rf test3.out test3.out2
        exit 0
    fi
fi
exit 1
