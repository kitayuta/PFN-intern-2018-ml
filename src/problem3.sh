#!/bin/bash

for e in "0.005" "0.010" "0.020" "0.050" "0.100" "0.200" "0.500"
do
    echo $e
    echo -n "random: "
    python random_sign.py $e
    python predict.py random_$e | tail -n 1
    echo -n "FGSM: "
    python FGSM.py $e
    python predict.py FGSM_$e | tail -n 1
done
