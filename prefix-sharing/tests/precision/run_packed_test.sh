#!/bin/bash
# Packed expand 精度测试
# TP=1, CP=1, 单卡, torchrun --nproc=1
# 注意: 测试脚本内部自行设置 Megatron args，不需要 GPT_ARGS 等参数

export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export TORCHDYNAMO_DISABLE=1
export PS_DEBUG=1

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6010
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
"

python3 -m torch.distributed.run $DISTRIBUTED_ARGS /tmp/prefix-sharing/tests/precision/test_packed_expand.py
