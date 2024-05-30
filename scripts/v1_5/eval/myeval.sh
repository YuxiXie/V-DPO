#!/bin/bash

IDX=0

CUDA_VISIBLE_DEVICES=$IDX python -m llava_dpo.eval.model_vqa \
    --model-path \
    /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/imp_sml/rlhfv-llava-dpo-noweight/checkpoint-1432 \
    --question-file \
    ./playground/data/amber_query.jsonl \
    --image-folder \
    /home/users/nus/e0672129/scratch/AMBER/data/image \
    --answers-file \
    /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/weight-rlhfv-llava-dpo-noweight-e4.jsonl

CUDA_VISIBLE_DEVICES=$IDX python -m llava_dpo.eval.model_vqa \
    --model-path \
    /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/imp_sml/rlhfv-llava-dpo-cfg/checkpoint-2860 \
    --question-file \
    ./playground/data/amber_query.jsonl \
    --image-folder \
    /home/users/nus/e0672129/scratch/AMBER/data/image \
    --answers-file \
    /home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/predictions/amber/weight-rlhfv-llava-dpo-cfg-e4.jsonl

# liuhaotian/llava-v1.5-7b
# /mnt/data/yuxi/llava/outputs/llava-v1.5-7b-dpo-myfilter/checkpoint-3072
# /mnt/data/yuxi/llava/outputs/llava-v1.5-7b-dpo-myfilter-img/checkpoint-3072
# /mnt/data/yuxi/llava/outputs/llava-v1.5-7b-dpo-myfilter-txt/checkpoint-8192

# OPENAI_API_KEY="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" python llava/eval/eval_gpt_review_visual.py \
#     --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
#     --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
#     --answer-list \
#     playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl \
#     /mnt/data/yuxi/dpo_llava/outputs/predictions/answer-file-sherlock.jsonl \
#     --rule llava/eval/table/rule.json \
#     --output \
#     /mnt/data/yuxi/dpo_llava/outputs/predictions/gpt4eval-file-sherlock.jsonl
