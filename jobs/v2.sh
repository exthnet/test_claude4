#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_v2
#PJM -o results/v2.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/v2_$(date +%Y%m%d_%H%M%S).log
echo "===== V2 (A pack hoisted) $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG

M=10000; N=10000; K=10000

echo "--- Correctness ---" | tee -a $LOG
AMX_MC=32 AMX_NC=512 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx 1024 1024 1024 3 1 1 2>&1 | tee -a $LOG

echo "--- KC scan @ MC=32 NC=512 PFD=0 ---" | tee -a $LOG
for KC in 256 384 512 640 768 1024 1280 1536 2048; do
  echo -n "KC=$KC : " | tee -a $LOG
  AMX_MC=32 AMX_NC=512 AMX_KC=$KC AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

echo "--- KC scan @ MC=32 NC=1024 PFD=0 ---" | tee -a $LOG
for KC in 512 768 1024 1280 1536 2048; do
  echo -n "KC=$KC : " | tee -a $LOG
  AMX_MC=32 AMX_NC=1024 AMX_KC=$KC AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

echo "--- MC scan @ NC=1024 KC=1024 PFD=0 ---" | tee -a $LOG
for MC in 32 64 96 128 192 256; do
  echo -n "MC=$MC : " | tee -a $LOG
  AMX_MC=$MC AMX_NC=1024 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- Final showdown @ 10000 ---" | tee -a $LOG
echo -n "AMX best : " | tee -a $LOG
AMX_MC=32 AMX_NC=1024 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
echo -n "MKL      : " | tee -a $LOG
$TASKSET $BIN mkl $M $N $K 5 2 0 2>&1 | tee -a $LOG

echo "===== V2 done $(date) =====" | tee -a $LOG
