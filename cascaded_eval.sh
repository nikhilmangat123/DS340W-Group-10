export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE="judgelm"
DATA_TYPE="judgelm"

# python3 -u src/cascaded_eval.py \
#     --data-type $DATA_TYPE \
#     --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
#     --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
#     --logit-file2 "relia_scores/qwen2.5/${DATA_TYPE}-logit.jsonl" \
#     --output-file2 "relia_scores/qwen2.5/${DATA_TYPE}-relia.json"

python3 -u src/cascaded_eval.py \
    --data-type $DATA_TYPE \
    --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
    --logit-file-gpt /root/autodl-tmp/UnlimitedJudge/outputs/${DATA_TYPE}-llama-3-70b-vanilla.jsonl