#/bin/bash


MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}
NODE_NUM=${NODE_NUM:-1}
NODE_RANK=${RANK:-0}
GPU_NUM_PER_NODE=${GPU_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}

HOOK_ACTIVE=20 HOOK_SKIP_FIRST=5 torchrun --nnodes=$NODE_NUM --nproc_per_node=$GPU_NUM_PER_NODE --node_rank $NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  imagenet.py ~/imagenet/imagenet/ILSVRC/Data/CLS-LOC --epochs 1 -a resnet50 --batch-size 256 --workers 8