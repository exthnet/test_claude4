#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_tune2
#PJM -o results/tune2.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/tune2_$(date +%Y%m%d_%H%M%S).log
echo "===== Tuning v2 $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG

M=10000; N=10000; K=10000

echo "--- Verify correctness ---" | tee -a $LOG
AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=4 $TASKSET $BIN amx 1024 1024 1024 3 1 1 2>&1 | tee -a $LOG

echo "--- PFD scan at MC=128 NC=1024 KC=10016 ---" | tee -a $LOG
for PFD in 0 2 4 6 8 12; do
  echo -n "PFD=$PFD : " | tee -a $LOG
  AMX_MC=128 AMX_NC=1024 AMX_KC=10016 AMX_PFD=$PFD $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- KC scan at MC=128 NC=1024 PFD=4 ---" | tee -a $LOG
for KC in 256 512 1024 2048 5120 10016; do
  echo -n "KC=$KC : " | tee -a $LOG
  AMX_MC=128 AMX_NC=1024 AMX_KC=$KC AMX_PFD=4 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- MC scan at KC=10016 NC=1024 PFD=4 ---" | tee -a $LOG
for MC in 32 64 96 128 192 256 384 512; do
  echo -n "MC=$MC : " | tee -a $LOG
  AMX_MC=$MC AMX_NC=1024 AMX_KC=10016 AMX_PFD=4 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- NC scan at MC=64 KC=10016 PFD=4 ---" | tee -a $LOG
for NC in 128 256 512 1024 2048 5120 10016; do
  echo -n "NC=$NC : " | tee -a $LOG
  AMX_MC=64 AMX_NC=$NC AMX_KC=10016 AMX_PFD=4 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "===== Tuning v2 done $(date) =====" | tee -a $LOG
