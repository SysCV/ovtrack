#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=$3

if [ -z "$3" ]; then
    PORT=29533
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4}
