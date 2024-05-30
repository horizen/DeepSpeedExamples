#/bin/bash

NODE_NUM=1
GPU_NUM_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=23456

NVSHMEM_NVTX=common nsys profile -t cuda,nvtx,osrt,cudnn,cublas --python-sampling-frequency=2000 --python-backtrace=cuda --python-sampling=true --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true -x true -w true -o ./nsys_profile torchrun --nnodes=$NODE_NUM --nproc_per_node=$GPU_NUM_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT synthetic_benchmark.py --model resnet50 --num-iters 10

