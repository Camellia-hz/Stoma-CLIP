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
    --save-frequency 5 \
    --report-to "tensorboard" \
    --train-data="data/single_symptoms_data.jsonl" \
    --val-data="data/single_symptoms_test.jsonl" \
    --csv-img-key image \
    --csv-caption-key caption \
    --warmup 10 \
    --batch-size=128 \
    --lr=1e-6 \
    --wd=0.1 \
    --epochs=500 \
    --workers=8 \
    --model RN50_fusion4 \
    --pretrained /mnt/csi-data-aly/user/haozhou/Projects/research/PMC-CLIP/ckpt/pmc_clip.pt \
    --log_dir ./logs/0320-Stoma-clip-train \
    --mlm \