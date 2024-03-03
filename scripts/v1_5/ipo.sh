#!/bin/bash

export WANDB_MODE=online
export WANDB_API_KEY="1396a7d2a29a8e8241dff6e0e6371f2ad61e11e2"

OUTPUT_DIR="/home/users/nus/e0672129/scratch/LLaVA-DPO/outputs/checkpoints/llava-v1.5-7b-ipo-sherlock-mix-365k-beta0.1"
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

deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT \
    --module llava_dpo.train.dpo_train \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --ipo \
    --scale_coeff 0.1 \
    --data_path /home/users/nus/e0672129/scratch/LLaVA-DPO/data/sherlock_filter/sherlock_similarityscores_mix-duplicate_365k.json \
    --image_folder /home/users/nus/e0672129/scratch/LLaVA-DPO/images/sherlock/train \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1024 \
    --save_total_limit 5 \
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
    --report_to wandb

# --tune_mm_mlp_adapter
# /mnt/data/yuxi/sherlock/img2img/train
# /mnt/data/guanzhen/coco/Image