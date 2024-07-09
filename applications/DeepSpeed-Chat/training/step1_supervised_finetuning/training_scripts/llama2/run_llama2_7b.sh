#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_llama2_7b
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT


MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-23456}
NODE_NUM=${NODE_NUM:-1}
NODE_RANK=${RANK:-0}
GPU_NUM_PER_NODE=${GPU_NUM_PER_NODE:-$(nvidia-smi -L | wc -l)}

torchrun --nnodes=$NODE_NUM --nproc_per_node=$GPU_NUM_PER_NODE --node_rank $NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
   training/step1_supervised_finetuning/main.py \
   --data_path ~/imagenet2/Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path ~/model/Llama-2-7b-hf \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --print_loss \
   --output_dir $OUTPUT