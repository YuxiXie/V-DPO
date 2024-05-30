#!/bin/bash

MODELNAME="rlhfv-llava-dpo-noweight"

MODEL_PATH="/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/imp_sml/${MODELNAME}/checkpoint-1432"

python -m llava_dpo.eval.model_vqa_loader \
    --model-path $MODEL_PATH --question-file ./playground/data/pope.jsonl \
    --image-folder /home/users/nus/e0672129/scratch/vqa/val2014 \
    --answers-file /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/predictions/pope/weight-${MODELNAME}-e4.jsonl \
    --temperature 0 --conv-mode vicuna_v1

bash /home/users/nus/e0672129/LLaVA-DPO/scripts/v1_5/eval/myeval.sh

# python llava_dpo/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl

# module load miniforge3/23.10
# conda activate llava-contrastive

# cd /home/users/nus/e0672129/scratch/VLMEvalKit

# bash run.sh
