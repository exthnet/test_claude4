#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_final
#PJM -o results/final.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/final_$(date +%Y%m%d_%H%M%S).log
echo "===== Final benchmark $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG
echo "CPU MHz: $(grep -m1 'cpu MHz' /proc/cpuinfo)" | tee -a $LOG

# === Correctness verification ===
echo "--- Correctness (1024x1024x1024) ---" | tee -a $LOG
AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx 1024 1024 1024 3 1 1 2>&1 | tee -a $LOG

# === Compare PFD light vs none at best KC ===
echo "--- PFD compare (10000x10000, KC=512, MC=128, NC=1024) ---" | tee -a $LOG
for PFD in 0 2 4 8; do
  echo -n "PFD=$PFD : " | tee -a $LOG
  AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=$PFD $TASKSET $BIN amx 10000 10000 10000 5 2 0 2>&1 | tee -a $LOG
done

# === KC sweep with PFD=0 (clean baseline) ===
echo "--- KC sweep PFD=0 (10000x10000, MC=128, NC=1024) ---" | tee -a $LOG
for KC in 256 384 512 768 1024 2048 4096 ; do
  echo -n "KC=$KC : " | tee -a $LOG
  AMX_MC=128 AMX_NC=1024 AMX_KC=$KC AMX_PFD=0 $TASKSET $BIN amx 10000 10000 10000 5 2 0 2>&1 | tee -a $LOG
done

# === MC NC scan at KC=512, PFD=0 ===
echo "--- MC scan (KC=512, NC=1024, PFD=0) ---" | tee -a $LOG
for MC in 32 64 96 128 192 256 ; do
  echo -n "MC=$MC : " | tee -a $LOG
  AMX_MC=$MC AMX_NC=1024 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx 10000 10000 10000 3 1 0 2>&1 | tee -a $LOG
done

echo "--- NC scan (KC=512, MC=128, PFD=0) ---" | tee -a $LOG
for NC in 256 512 1024 2048 5120 10016 ; do
  echo -n "NC=$NC : " | tee -a $LOG
  AMX_MC=128 AMX_NC=$NC AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx 10000 10000 10000 3 1 0 2>&1 | tee -a $LOG
done

# === All-methods comparison at 10000 ===
echo "--- All methods @ 10000x10000x10000 ---" | tee -a $LOG
AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN avx512 10000 10000 10000 3 1 0 2>&1 | tee -a $LOG
AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx 10000 10000 10000 5 2 0 2>&1 | tee -a $LOG
AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN mkl 10000 10000 10000 5 2 0 2>&1 | tee -a $LOG

# === Size scan ===
echo "--- Size scan AMX vs MKL (best params) ---" | tee -a $LOG
for sz in 256 512 1024 2048 4096 6144 8192 10000; do
  echo -n "AMX sz=$sz : " | tee -a $LOG
  AMX_MC=128 AMX_NC=1024 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx $sz $sz $sz 3 1 0 2>&1 | tee -a $LOG
  echo -n "MKL sz=$sz : " | tee -a $LOG
  $TASKSET $BIN mkl $sz $sz $sz 3 1 0 2>&1 | tee -a $LOG
done

# Naive only at small sizes (for context)
echo "--- Naive baseline (small only) ---" | tee -a $LOG
$TASKSET $BIN naive 256 256 256 2 1 1 2>&1 | tee -a $LOG
$TASKSET $BIN naive 512 512 512 1 1 0 2>&1 | tee -a $LOG

echo "===== Done $(date) =====" | tee -a $LOG
