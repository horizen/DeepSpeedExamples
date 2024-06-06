#/bin/bash

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}
NODE_NUM=${NODE_NUM:-1}
NODE_RANK=${RANK:-0}
GPU_NUM_PER_NODE=${GPU_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}

NCCL_DEBUG=INFO NVSHMEM_NVTX=common nsys profile -t cuda,nvtx,osrt --python-sampling-frequency=2000 --python-backtrace=cuda --python-sampling=true --capture-range=cudaProfilerApi -s cpu --cudabacktrace=true -x true -w true -o ./nsys_profile python -m torch.distributed.run --nnodes=$NODE_NUM --nproc_per_node=$GPU_NUM_PER_NODE --node_rank $NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT synthetic_benchmark.py --model resnet50 --num-iters 10