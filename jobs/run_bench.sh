#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_bench
#PJM -o results/run_bench.out

set -e
cd $PJM_O_WORKDIR

source /etc/profile.d/modules.sh
module load gcc-toolset/13

BIN=./bench
LOG=results/bench_$(date +%Y%m%d_%H%M%S).log
echo "===== Job started $(date) =====" | tee $LOG
echo "hostname: $(hostname)" | tee -a $LOG
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1)" | tee -a $LOG
echo "CPU MHz: $(grep 'cpu MHz' /proc/cpuinfo | head -1)" | tee -a $LOG
echo "AMX flags: $(grep -E -o 'amx_(tile|bf16|int8)' /proc/cpuinfo | sort -u | tr '\n' ' ')" | tee -a $LOG

# Bind to core 0 for single-core measurement
TASKSET="taskset -c 0"

run() {
    local meth=$1 M=$2 N=$3 K=$4 it=$5 wu=$6 ver=$7
    echo "---- $meth $M $N $K iters=$it warmups=$wu verify=$ver ----" | tee -a $LOG
    $TASKSET $BIN $meth $M $N $K $it $wu $ver 2>&1 | tee -a $LOG
}

# Correctness verification at small sizes
echo "=== Correctness verification ===" | tee -a $LOG
run all 256 256 256 2 1 1

# Larger correctness check (skip naive)
echo "=== Larger size correctness (skip naive) ===" | tee -a $LOG
run avx512 1024 1024 1024 3 1 1
run amx    1024 1024 1024 3 1 1
run mkl    1024 1024 1024 3 1 1

# Main benchmark @ 10000x10000x10000
echo "=== Main benchmark @ 10000 ===" | tee -a $LOG
run avx512 10000 10000 10000 5 2 0
run amx    10000 10000 10000 5 2 0
run mkl    10000 10000 10000 5 2 0

# Scan a few sizes for AMX
echo "=== AMX size scan ===" | tee -a $LOG
for sz in 512 1024 2048 4096 6144 8192 10000; do
    run amx $sz $sz $sz 3 1 0
done

echo "=== Job ended $(date) ===" | tee -a $LOG
