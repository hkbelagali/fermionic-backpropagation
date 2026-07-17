which python
python -c "import sys; print(sys.executable)"

NUM_PARALLEL=4
count=0

for i in 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00
    do
        (
            cd ${i}
            cp ../run_UCJ_no_opt.py ./
            sed -i "s/XXXX/${i}/g" run_UCJ_no_opt.py
            python run_UCJ_no_opt.py
        ) &
        count=$((count + 1))
        if (( count % NUM_PARALLEL == 0 )); then
            wait
        fi
    done
wait
