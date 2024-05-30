#!/bin/bash

MODEL="weight-llava-dpo-cfg-e4"

python llava_dpo/eval/eval_pope.py \
    --annotation-dir ./playground/data/pope \
    --question-file ./playground/data/pope.jsonl \
    --result-file /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/$MODEL.jsonl \
    > /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/$MODEL.log

# MODEL="nolb-lb-llava-cdpo-wl-sl-e1"

# python llava_dpo/eval/eval_pope.py \
#     --annotation-dir ./playground/data/pope \
#     --question-file ./playground/data/pope.jsonl \
#     --result-file /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/$MODEL.jsonl \
#     > /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/$MODEL.log

# MODEL="llava-cdpo-e8"

# python llava_dpo/eval/eval_pope.py \
#     --annotation-dir ./playground/data/pope \
#     --question-file ./playground/data/pope.jsonl \
#     --result-file /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/$MODEL.jsonl \
#     > /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/$MODEL.log

# MODEL="rlhfv-dpo-nolb-e8"

# python llava_dpo/eval/eval_pope.py \
#     --annotation-dir ./playground/data/pope \
#     --question-file ./playground/data/pope.jsonl \
#     --result-file /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/pope/$MODEL.jsonl \
#     > /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/pope/$MODEL.log