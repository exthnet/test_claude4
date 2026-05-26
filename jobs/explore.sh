#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_exp
#PJM -o results/explore.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/explore_$(date +%Y%m%d_%H%M%S).log
echo "===== Explore $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG

M=10000; N=10000; K=10000

# Sanity check at best params
echo "--- Sanity (best) ---" | tee -a $LOG
AMX_MC=32 AMX_NC=512 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG

# MC scan at NC=512 KC=1024 (v2 best)
echo "--- MC scan (NC=512, KC=1024) ---" | tee -a $LOG
for MC in 32 64 96 128 192 256 ; do
  echo -n "MC=$MC : " | tee -a $LOG
  AMX_MC=$MC AMX_NC=512 AMX_KC=1024 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

# (NC, KC) grid around the sweet spot
echo "--- (NC, KC) grid ---" | tee -a $LOG
for NC in 384 512 640 768 ; do
  for KC in 768 896 1024 1152 1280 ; do
    echo -n "NC=$NC KC=$KC : " | tee -a $LOG
    AMX_MC=32 AMX_NC=$NC AMX_KC=$KC AMX_PFD=0 $TASKSET $BIN amx $M $N $K 3 1 0 2>&1 | tee -a $LOG
  done
done

echo "===== Done $(date) =====" | tee -a $LOG
