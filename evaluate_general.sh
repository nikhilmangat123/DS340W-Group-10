#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATA_TYPE=pandalm
python3 -u src/evaluate_judge.py \
    --model-name-or-path "/root/autodl-tmp/models/Meta-Llama-3-70B-Instruct" \
    --prompt-type "vanilla" \
    --model-type "general" \
    --data-type $DATA_TYPE \
    --max-new-token 1024 \
    --logit-file "/root/autodl-tmp/UnlimitedJudge/relia_scores/llama3-70B/${DATA_TYPE}-logit.jsonl" \
    --apply-chat-template