#!/bin/bash

LOG_DIR="/home/users/nus/e0672129/scratch/AMBER/log"
mkdir -p $LOG_DIR
exec 1> >(tee "${LOG_DIR}/stdout.log" >&1) 2> >(tee "${LOG_DIR}/stderr.log" >&2)

OUTPUT_DIR="/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber"

FNAME1="nolb-llava-cdpo-e4"
FNAME2="nolb-llava-cdpo-wl-e4"
FNAME3="nolb-llava-cdpo-lbwl-e4"
FNAME4="nolb-llava-cdpo-lbwl-sl-e4"
FNAME5="nolb-llava-cdpo-dylbwl-e4"
FNAME6="nolb-llava-cdpo-dylbwl-sl-e4"

for FNAME in $FNAME1 $FNAME2 $FNAME3 $FNAME4 $FNAME5 $FNAME6
do
    python inference.py \
        --inference_data ${OUTPUT_DIR}/${FNAME}_response.json \
        --evaluation_type a \
        > ${OUTPUT_DIR}/${FNAME}.log
done
