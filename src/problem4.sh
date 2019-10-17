#!/bin/bash

for s in 1 2 4 8 16 32
do
    echo $s
    echo -n "random_binary: "
    python random_binary.py $s
    python predict.py random_binary_$s | tail -n 1
    echo -n "FGSM_binary: "
    python FGSM_binary.py $s
    python predict.py FGSM_binary_$s | tail -n 1
done
