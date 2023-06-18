#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NAME=$4
COMMAND=$5
TIME=$6
PORT=$7
MEM=$8
CORE=$9
GPUN=${10}


LOG=".log"

if [ -z "$6" ]; then
    TIME=4
fi

if [ -z "${10}" ]; then
    GPUN="NVIDIAGeForceRTX2080Ti"
fi

echo bsub -n ${CORE} -W $TIME:00 -J "${NAME}" -oo "${NAME}${LOG}" \
    -R "rusage[mem=${MEM},ngpus_excl_p=${GPUS}]" \
    -R "select[gpu_model0==${GPUN}]" \
    "./tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} ${PORT} ${COMMAND}"

bsub -n ${CORE} -W $TIME:00 -J $NAME -oo $NAME$LOG \
    -R "rusage[mem=${MEM}, ngpus_excl_p=${GPUS}]" \
    -R "select[gpu_model0==${GPUN}]" \
    "./tools/dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS} ${PORT} ${COMMAND}"
