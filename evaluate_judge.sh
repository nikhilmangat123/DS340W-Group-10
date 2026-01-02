#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL_TYPE=judgelm
DATA_TYPE=judgelm
python3 -u src/evaluate_judge.py \
    --model-name-or-path "/root/autodl-tmp/models/JudgeLM-7B-v1.0" \
    --prompt-type "vanilla" \
    --model-type "judgelm" \
    --data-type "judgelm" \
    --max-new-token 1024 \
    --logit-file "/root/autodl-tmp/UnlimitedJudge/relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \