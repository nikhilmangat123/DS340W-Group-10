export CUDA_VISIBLE_DEVICES=0

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE="auto-j"
DATA_TYPE=judgelm

# python3 -u src/cal_reliability.py \
#     --model-name-or-path "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" \
#     --model-type $MODEL_TYPE \
#     --data-type $DATA_TYPE \
#     --max-new-token 1024 \
#     --logit-file "relia_scores/qwen2.5/${DATA_TYPE}-logit.jsonl" \
#     --output-file "relia_scores/qwen2.5/${DATA_TYPE}-relia.json" \
#     --apply-chat-template

python3 -u src/cal_reliability.py \
    --model-name-or-path "/root/autodl-tmp/models/Auto-J-13B" \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --max-new-token 1024 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"