$PLUSPY_EXEC -S0 -c100 -i Spec TestFIFO > test9.out 2>test9.out2
if cmp -s test9.out tests/regression/test9.exp
then
    if cmp -s test9.out2 tests/regression/test9.exp2
    then
        rm -rf test9.out test9.out2
        exit 0
    fi
fi
exit 1
