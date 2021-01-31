set a="1"
set lr="0.05"
set t="250"
set n="100"
set L="30"

@REM python Ising1D_RBMSym_order.py -L %L% -g 0.0 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 0.2 -a %a% -lr %lr% -log "logfile.txt" -n 1000 -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 0.4 -a %a% -lr %lr% -log "logfile.txt" -n 500 -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 0.6 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 0.8 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 1.0 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 1.2 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 1.4 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 1.6 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 1.8 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%
python Ising1D_RBMSym_order.py -L %L% -g 2.0 -a %a% -lr %lr% -log "logfile.txt" -n %n% -t %t%



