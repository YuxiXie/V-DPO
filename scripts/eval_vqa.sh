#!/bin/bash

IDX=1

# CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa \
#     --model-path \
#     /mnt/data/yuxi/llava/outputs/llava-v1.5-7b-dpo-test/checkpoint-1000 \
#     --question-file \
#     playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
#     --image-folder \
#     /mnt/data/yuxi/coco/coco2014/val2014 \
#     --answers-file \
#     /mnt/data/yuxi/dpo_llava/outputs/predictions/answer-file-sherlock.jsonl

CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa \
    --model-path \
    liuhaotian/llava-v1.5-7b \
    --question-file \
    /mnt/data/yuxi/shikra/data/coco_pope_popular.jsonl \
    --image-folder \
    /mnt/data/yuxi/coco/coco2014/val2014 \
    --answers-file \
    /mnt/data/yuxi/dpo_llava/outputs/predictions/pope-popular-answer-file-baseline.jsonl

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
