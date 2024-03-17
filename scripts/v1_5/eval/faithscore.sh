#!/bin/bash

EXE_HOME="/home/yuxi/Projects/LLaVA-DPO"


export CUDA_VISIBLE_DEVICES=0

# python $EXE_HOME/scripts/faithscore_eval.py \
#     qa-ptx-vcrcoco-27k-noptx-e1.jsonl \
#     qa-ptx-vcrcoco-27k-e1.jsonl \
#     qa-ptx-coco-14k-e2.jsonl \
#     qa-ptx-vcrcoco-27k-e2.jsonl \
#     qa-ptx-vcrcoco-27k-noptx-e2.jsonl


python $EXE_HOME/scripts/faithscore_eval.py qa-ptx-vcrcoco-27k-noptx-e1.jsonl

python $EXE_HOME/scripts/faithscore_eval.py qa-ptx-vcrcoco-27k-e1.jsonl

python $EXE_HOME/scripts/faithscore_eval.py qa-ptx-coco-14k-e2.jsonl

python $EXE_HOME/scripts/faithscore_eval.py qa-ptx-vcrcoco-27k-e2.jsonl

python $EXE_HOME/scripts/faithscore_eval.py qa-ptx-vcrcoco-27k-noptx-e2.jsonl
