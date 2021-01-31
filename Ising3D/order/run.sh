#!/bin/bash

for g in 3.75 4.25 4.75 5.25 5.75 6.25
do
    python Ising3D_RBMSym_order.py -L1 4 -L2 4 -L3 4 -g $g -log logfile.txt -lr 0.05 -a 2 -t 300 -n 100
done

for g in 2.0 3.0 3.5
do
    python Ising3D_RBMSym_order.py -L1 4 -L2 4 -L3 4 -g $g -log logfile.txt -lr 0.075 -a 4 -t 300 -n 100
done