#!/bin/sh
accelerate launch --mixed_precision="fp16" --config_file=config_accelerate.yaml  map_generation/train_text_to_image.py \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="result_path" \
    --max_train_steps=4 \
    --dataset_name=data