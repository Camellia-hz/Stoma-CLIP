#!/bin/bash

set -x

if [ -z "$1" ]; then
    NPROC=8
else
    NPROC=$1
fi

torchrun --nproc_per_node=$NPROC \
    training/main.py \
    --dataset-type "csv" \
    --image_dir ./data/cleaned_data \
    --csv-separator "," \
    --save-frequency 10 \
    --report-to "tensorboard" \
    --train-data="data/single_symptoms_train.jsonl" \
    --csv-img-key image \
    --csv-caption-key caption \
    --warmup 10 \
    --batch-size=256 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs=200 \
    --workers=8 \
    --model RN50_fusion4 \
    --pretrained ckpt/pmc_clip.pt \
    --log_dir ./logs/0322-pmc-clip-train-cls-cross-attention\
    --fusion_method "cross_attention" \
    --mlm \