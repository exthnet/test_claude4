#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_f2
#PJM -o results/final2.out

set -e
cd $PJM_O_WORKDIR
source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
TASKSET="taskset -c 0"
LOG=results/final2_$(date +%Y%m%d_%H%M%S).log
echo "===== Final2 $(date) =====" | tee $LOG
echo "host: $(hostname)" | tee -a $LOG

M=10000; N=10000; K=10000

# Correctness at small size with best params
echo "--- Correctness (1024x1024x1024, MC=32, NC=512, KC=512, PFD=0) ---" | tee -a $LOG
AMX_MC=32 AMX_NC=512 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx 1024 1024 1024 3 1 1 2>&1 | tee -a $LOG

# Fine scan with MC=32
echo "--- MC=32 NC scan KC=512 PFD=0 ---" | tee -a $LOG
for NC in 256 512 768 1024 1536 2048; do
  echo -n "MC=32 NC=$NC : " | tee -a $LOG
  AMX_MC=32 AMX_NC=$NC AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

echo "--- MC=32 KC scan NC=512 PFD=0 ---" | tee -a $LOG
for KC in 256 384 512 640 768 1024; do
  echo -n "MC=32 NC=512 KC=$KC : " | tee -a $LOG
  AMX_MC=32 AMX_NC=512 AMX_KC=$KC AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
done

# Final comparison with best params: MC=32, NC=512, KC=512
echo "--- Final comparison @ 10000x10000x10000 ---" | tee -a $LOG
echo -n "AVX-512 BF16 : " | tee -a $LOG
AMX_MC=32 AMX_NC=512 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN avx512 $M $N $K 3 1 0 2>&1 | tee -a $LOG
echo -n "AMX (best)   : " | tee -a $LOG
AMX_MC=32 AMX_NC=512 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx $M $N $K 5 2 0 2>&1 | tee -a $LOG
echo -n "MKL          : " | tee -a $LOG
$TASKSET $BIN mkl $M $N $K 5 2 0 2>&1 | tee -a $LOG

# Size scan with best params
echo "--- Size scan AMX (best params) vs MKL ---" | tee -a $LOG
for sz in 512 1024 2048 4096 6144 8192 10000; do
  echo -n "AMX sz=$sz : " | tee -a $LOG
  AMX_MC=32 AMX_NC=512 AMX_KC=512 AMX_PFD=0 $TASKSET $BIN amx $sz $sz $sz 3 1 0 2>&1 | tee -a $LOG
  echo -n "MKL sz=$sz : " | tee -a $LOG
  $TASKSET $BIN mkl $sz $sz $sz 3 1 0 2>&1 | tee -a $LOG
done

echo "===== Done $(date) =====" | tee -a $LOG
