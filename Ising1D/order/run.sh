#!/bin/bash

for g in 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0
do
    python Ising1D_RBMSym_order.py -L 4 -g $g -log logfile.txt -lr 0.05 -a 1 -t 500 -n 100 -ed
done

for g in 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0
do
    python Ising1D_RBMSym_order.py -L 8 -g $g -log logfile.txt -lr 0.05 -a 1 -t 500 -n 100 -ed
done

for g in 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0
do
    python Ising1D_RBMSym_order.py -L 30 -g $g -log logfile.txt -lr 0.05 -a 1 -t 500 -n 100
done

for g in 0.2 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0
do
    python Ising1D_RBMSym_order.py -L 100 -g $g -log logfile.txt -lr 0.05 -a 1 -t 500 -n 100
done