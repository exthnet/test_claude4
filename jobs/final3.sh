#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_f3
#PJM -o results/final3.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/final3_$(date +%Y%m%d_%H%M%S).log
echo "===== Final3 $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG
echo "CPU MHz: $(grep -m1 'cpu MHz' /proc/cpuinfo)" | tee -a $LOG

M=10000; N=10000; K=10000

echo "--- Correctness @ 1024 ---" | tee -a $LOG
AMX_MC=32 AMX_NC=512 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx 1024 1024 1024 3 1 1 2>&1 | tee -a $LOG

# Best params from v2 scan: MC=32 NC=512 KC=1024 PFD=0
echo "--- Best AMX @ 10000 (MC=32 NC=512 KC=1024) ---" | tee -a $LOG
for i in 1 2 3; do
    echo -n "Run $i: " | tee -a $LOG
    AMX_MC=32 AMX_NC=512 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

echo "--- MKL @ 10000 ---" | tee -a $LOG
for i in 1 2 3; do
    echo -n "Run $i: " | tee -a $LOG
    $TASKSET $BIN mkl $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

echo "--- Size scan AMX (best params) vs MKL ---" | tee -a $LOG
for sz in 512 1024 2048 4096 6144 8192 10000; do
  echo -n "AMX sz=$sz : " | tee -a $LOG
  AMX_MC=32 AMX_NC=512 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx $sz $sz $sz 3 1 0 2>&1 | tee -a $LOG
  echo -n "MKL sz=$sz : " | tee -a $LOG
  $TASKSET $BIN mkl $sz $sz $sz 3 1 0 2>&1 | tee -a $LOG
done

echo "===== Done $(date) =====" | tee -a $LOG
