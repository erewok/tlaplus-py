$PLUSPY_EXEC -S0 -c100 Qsort > test7.out 2>test7.out2
if cmp -s test7.out tests/regression/test7.exp
then
    if cmp -s test7.out2 tests/regression/test7.exp2
    then
        rm -rf test7.out test7.out2
        exit 0
    fi
fi
exit 1
