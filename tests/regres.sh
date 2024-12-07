export PLUSPY_EXEC=../pluspy.sh
export PLUSPYPATH="..:../modules/lib:../modules/book:../modules/other"

for i in 1 2 3 4 5 6 7 9 10 11
do
    echo running test $i
    sh regression/test$i.sh
    case $? in
    0)
        ;;
    *)
        echo test $i failed
        exit 1
    esac
done
