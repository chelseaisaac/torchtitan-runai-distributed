#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

export NCCL_DEBUG=INFO
export NCCL_IGNORE_CPU_AFFINITY=1

master_addr=$MASTER_ADDR
master_port=$MASTER_PORT

# nnode override
if [ "$OVERRIDE_PET_NNODES" = "" ]; then
    job_n=$PET_NNODES
else
    job_n=$OVERRIDE_PET_NNODES
fi

# Use envs as local overrides for convenience e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}
RDZV_TIMEOUT=${RDZV_TIMEOUT:-"600"}
MAX_RESTARTS=${MAX_RESTARTS:-1}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --nnodes=${job_n} \
--max-restarts ${MAX_RESTARTS} --rdzv_backend c10d \
--rdzv_conf timeout=${RDZV_TIMEOUT} --rdzv_endpoint=${master_addr}:${master_port} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
