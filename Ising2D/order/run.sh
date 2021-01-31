#!/bin/bash

for g in 2.6 2.7 2.8 2.9 3.1 3.2 3.3 3.4 4.5 5.0
do
    python Ising2D_RBMSym_order.py -L1 6 -L2 6 -g $g -log logfile.txt -lr 0.05 -a 2 -t 300 -n 100
done

for g in 1.0 1.5 2.0 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 4.0 4.5 5.0
do
    python Ising2D_RBMSym_order.py -L1 8 -L2 8 -g $g -log logfile.txt -lr 0.05 -a 2 -t 300 -n 100
done

for g in 1.0 1.5 2.0 2.5 2.6 2.7 2.8 2.9 3.0 3.1 3.2 3.3 3.4 3.5 4.0 4.5 5.0
do
    python Ising2D_RBMSym_order.py -L1 10 -L2 10 -g $g -log logfile.txt -lr 0.05 -a 2 -t 300 -n 100
done