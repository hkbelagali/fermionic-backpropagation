which python
python -c "import sys; print(sys.executable)"
python -c "import jax; print(jax.__version__)"
python -c "import jax; print(jax.devices())"

NUM_GPUS=4
count=0

for i in 0.70 0.80 0.90 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00
    do
        gpu=$((count % NUM_GPUS))
        (
            cd ${i}
            cp ../run_UCJ.py ./
            sed -i "s/XXXX/${i}/g" run_UCJ.py
           #  CUDA_VISIBLE_DEVICES=${gpu} python run_UCJ.py
            cp ../run_HCI.py ./
            sed -i "s/XXXX/${i}/g" run_HCI.py
            CUDA_VISIBLE_DEVICES=${gpu} python run_HCI.py
        ) &
        count=$((count + 1))
        if (( count % NUM_GPUS == 0 )); then
            wait
        fi
    done
wait
 
