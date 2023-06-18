#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NAME=$3
COMMAND=$4
TIME=$5
PORT=$6
MEM=$7
CORE=$8
GPUN=$9

FOLDER="logs/"
LOG=".log"

if [ -z "$6" ]; then
    TIME=4
fi

if [ -z "$9" ]; then
    GPUN="NVIDIAGeForceRTX2080Ti"
fi

echo bsub -n ${CORE} -W $TIME:00 -J $NAME -oo $NAME$LOG \
    -R "rusage[mem=${MEM},ngpus_excl_p=${GPUS}]" \
    -R "select[gpu_model0==${GPUN}]" \
    "./tools/dist_train.sh ${CONFIG} ${GPUS} ${PORT} ${COMMAND}"

bsub -n ${CORE} -W $TIME:00 -J $NAME -oo $NAME$LOG \
    -R "rusage[mem=${MEM},ngpus_excl_p=${GPUS}]" \
    -R "select[gpu_model0==${GPUN}]" \
    "./tools/dist_train.sh ${CONFIG} ${GPUS} ${PORT} ${COMMAND}"

