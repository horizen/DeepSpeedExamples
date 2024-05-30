#/bin/bash

dlprof --mode pytorch torchrun --nnodes=$NODE_NUM --nproc-per-node=$GPU_NUM_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT synthetic_benchmark.py --model resnet50 --num-iters 10