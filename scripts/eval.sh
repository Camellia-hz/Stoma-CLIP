
python training/main.py \
    --dataset-type "csv" --csv-separator "," --report-to tensorboard \
    --val-data="data/pmc_oa/test.jsonl" \
    --csv-img-key image --csv-caption-key caption \
    --batch-size=32 --workers=8 \
    --model RN50_fusion4 --mlm --crop-scale 0.1 \
    --resume ckpt/pmc_clip.pt \
    --test-2000 \
    --image_dir ./data/pmc_oa/caption_T060_filtered_top4_sep_v0_subfigures