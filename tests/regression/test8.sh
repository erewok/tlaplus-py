$PLUSPY_EXEC -S0 -c100 TestBinBosco > test8.out 2>test8.out2
if cmp -s test8.out tests/regression/test8.exp
then
    if cmp -s test8.out2 tests/regression/test8.exp2
    then
        rm -rf test8.out test8.out2
        exit 0
    fi
fi
exit 1
