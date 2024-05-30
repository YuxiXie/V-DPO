#!/bin/bash

export WANDB_MODE=online
export WANDB_API_KEY="1396a7d2a29a8e8241dff6e0e6371f2ad61e11e2"

OUTPUT_DIR="/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/experiments/llava/imp_sml/llava-dpo-cfgref"
mkdir -p $OUTPUT_DIR

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

cp -f "$0" "${OUTPUT_DIR}/script.sh"

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P

gpu_vis=0,1

MODEL_PATH="liuhaotian/llava-v1.5-7b"
REF_MODEL_PATH="liuhaotian/llava-v1.5-7b"

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
    --module llava_dpo.train.dpo_train \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path $MODEL_PATH \
    --ref_model_name_or_path $REF_MODEL_PATH \
    --n_random_images 0 \
    --lbwl \
    --gamma 2.0 \
    --version v1 \
    --scale_coeff 0.1 \
    --data_path /home/users/nus/e0672129/scratch/LLaVA-DPO/data/all_preferences_5k_v10.json \
    --image_folder /home/users/nus/e0672129/scratch \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2048 \
    --learning_rate 1e-6 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --log_project LLaVA-DPO-WL \
    --report_to wandb

# bash scripts/v1_5/dpo.sh