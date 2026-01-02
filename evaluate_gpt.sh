#!/bin/bash
python3 -u src/evaluate_gpt.py \
    --model-name "llama-3-70b" \
    --prompt-type "vanilla" \
    --data-type "judgelm" \
    --multi-process True \
    --max-new-token 4096 