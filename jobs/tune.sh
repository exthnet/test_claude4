#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_tune
#PJM -o results/tune.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/tune_$(date +%Y%m%d_%H%M%S).log
echo "===== Tuning $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG

# Tuning grid for 10000x10000
M=10000; N=10000; K=10000
echo "--- Scan KC at MC=128 NC=1024 ---" | tee -a $LOG
for KC in 256 512 1024 2048 5120 10016; do
  echo -n "MC=128 NC=1024 KC=$KC : " | tee -a $LOG
  AMX_MC=128 AMX_NC=1024 AMX_KC=$KC $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- Scan KC at MC=64 NC=1024 ---" | tee -a $LOG
for KC in 1024 2048 5120 10016; do
  echo -n "MC=64 NC=1024 KC=$KC : " | tee -a $LOG
  AMX_MC=64 AMX_NC=1024 AMX_KC=$KC $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- Scan KC at MC=128 NC=2048 ---" | tee -a $LOG
for KC in 1024 2048 5120 10016; do
  echo -n "MC=128 NC=2048 KC=$KC : " | tee -a $LOG
  AMX_MC=128 AMX_NC=2048 AMX_KC=$KC $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- Scan MC at NC=512, KC=10016 ---" | tee -a $LOG
for MC in 32 64 128 256 512; do
  echo -n "MC=$MC NC=512 KC=10016 : " | tee -a $LOG
  AMX_MC=$MC AMX_NC=512 AMX_KC=10016 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "--- Scan NC at MC=64, KC=10016 ---" | tee -a $LOG
for NC in 128 256 512 1024 2048 5120 10016; do
  echo -n "MC=64 NC=$NC KC=10016 : " | tee -a $LOG
  AMX_MC=64 AMX_NC=$NC AMX_KC=10016 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
done

echo "===== Tuning done $(date) =====" | tee -a $LOG
