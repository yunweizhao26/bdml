#!/bin/bash

# Usage:
# ./run_deepspeed.sh

# Make sure path to the model is correct
MODEL_PATH="./models/Llama3.2-3B"
OUTPUT_DIR="./llama-climate-deepspeed"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run DeepSpeed with pipeline parallelism
deepspeed --num_gpus=2 \
    train_pp_2.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --num_stages 8 \
    --fp16 \
    --zero_stage 1 \
    --batch_size 8 \
    --epochs 5 \
    --deepspeed \
    --config ds_config.json