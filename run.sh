#!/bin/sh
accelerate launch --mixed_precision="fp16" --config_file=config_accelerate.yaml  map_generation/train_text_to_image.py \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="result_path" \
    --snr_gamma=5.0 \
    --p_uncond=0.1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --use_ema