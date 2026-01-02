from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import os
import gc
import json
import vllm
# 假设这些是你本地的模块
from build_prompt_judge import create_prompt, parse_predictions
from build_dataset import build_dataset, calculate_metrics
from evaluate_judge import build_params

# ... (get_multi_answer 和 get_batch_evaluation 函数保持不变，此处省略以节省篇幅) ...
# 请保留原来代码中的 get_multi_answer 和 get_batch_evaluation 定义

@torch.inference_mode()
def get_multi_answer(
    model_path,
    prompts,
    max_new_token=2048,
    temperature=0.1,
    top_p=1.0,
    apply_chat_template=False,
):
    print("Start load VLLM model!")
    # 注意：如果显存紧张，可以适当降低 gpu_memory_utilization
    model = vllm.LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), dtype="bfloat16", gpu_memory_utilization=0.6)
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_new_token,
        top_p=top_p,
    )
    print("VLLM model loaded!")

    tokenizer = model.get_tokenizer()
    # 这里的 512 是预留给 prompt 的，防止溢出
    MAX_LEN = model.llm_engine.model_config.max_model_len - 512
    
    if apply_chat_template:
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
    
    # 截断 prompt 以防过长
    prompt_ids = [tokenizer.encode(prompt)[-MAX_LEN:] for prompt in prompts]
    
    # VLLM 生成
    pred_list = model.generate(prompts=[vllm.TokensPrompt({"prompt_token_ids": p}) for p in prompt_ids], sampling_params=sampling_params)

    prompt_token_ids = [it.prompt_token_ids for it in pred_list]
    output_token_ids = [it.outputs[0].token_ids for it in pred_list]

    prefix_lens = [len(p_ids) for p_ids in prompt_token_ids]
    target_lens = [len(o_ids) for o_ids in output_token_ids]

    output_tokens = [it.outputs[0].text for it in pred_list]

    # 合并 prompt 和 output ids
    output_ids = [ids[0]+ids[1] for ids in zip(prompt_token_ids, output_token_ids)]

    return output_tokens, prefix_lens, target_lens, output_ids


@torch.inference_mode()
def get_batch_evaluation(
    model,
    batch_input_ids,  # List[List[int]]
    batch_prefix_lens, # List[int]
    batch_target_lens, # List[int]
    pad_token_id=0
):
    # ... (保持原有的 get_batch_evaluation 代码不变) ...
    """
    批量计算 Entropy。
    """
    device = model.device
    batch_size = len(batch_input_ids)
    
    # 1. 找出当前 batch 中的最大长度
    max_len = max(len(ids) for ids in batch_input_ids)
    
    # 2. 初始化 Tensor
    # input_ids 使用 pad_token_id 填充
    input_tensor = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    # labels 使用 -100 填充 (忽略计算 loss 的部分)
    labels_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    # attention_mask
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    
    # 3. 填充数据
    for i, (ids, pre_len) in enumerate(zip(batch_input_ids, batch_prefix_lens)):
        seq_len = len(ids)
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        
        input_tensor[i, :seq_len] = ids_tensor
        attention_mask[i, :seq_len] = 1
        
        # 复制到 labels
        labels_tensor[i, :seq_len] = ids_tensor
        # Mask 掉 instruction 部分
        labels_tensor[i, :pre_len] = -100
        # 此时 Padding 部分已经是 -100 了
        
    # 4. 模型前向传播
    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    
    logits = outputs.logits # (B, L, V)
    logprobs = F.log_softmax(logits, dim=-1) # (B, L, V)
    
    # 5. 复现原本的 Metric 计算逻辑
    # 创建 mask: (B, L) True where labels is -100 (Instruction or Padding)
    mask = (labels_tensor == -100)
    
    # 将 mask 扩展到 logprobs 的维度或利用广播将对应位置置 0
    logprobs[mask] = 0.0
    
    # 计算 entropy term: (B, L, V) -> mean -> (B, L)
    logprobs_entropy = torch.mean(logprobs * logits, dim=-1)
    
    # 对序列长度求和
    sum_entropy = logprobs_entropy.sum(dim=-1) # (B,)
    
    # 除以 target_len
    target_lens_tensor = torch.tensor(batch_target_lens, device=device, dtype=torch.float)
    evaluation_ents = sum_entropy / target_lens_tensor
    
    return [{"entropy": ent.item()} for ent in evaluation_ents]


if __name__ == "__main__":
    parser = build_params()
    parser.add_argument("--cali-model-name-or-path", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size for reliability scoring")
    
    args = parser.parse_args()
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = build_dataset(args.data_type)
    print(f"Loaded dataset from {args.data_path}")
    print(f"The length is {len(dataset)}")

    instruction = create_prompt(args.model_type, args.data_type)

    prompts = []
    answers = []
    for index, example in enumerate(dataset):
        if args.model_type in ["judgelm", "pandalm", "auto-j", "llama-3"]:
            if args.data_type in ["prometheus-ind", "prometheus-ood", "toxic-chat", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa"]:
                prompt = instruction.format(question_body=example["question_body"],
                                            answer_body=example["answer_body"])
                prompts.append(prompt)
            else:
                example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
                prompts.append(prompt)

        elif args.model_type == "prometheus":
            if args.data_type in ["prometheus-ind", "prometheus-ood", "toxic-chat", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa"]:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer_body=example["answer_body"])
                prompts.append(prompt)
            else:
                example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
                prompt_a = instruction.format(question_body=example["question_body"],
                                                rubric=example["rubric"],
                                                answer_body=example["answer1_body"])
                prompt_b = instruction.format(question_body=example["question_body"],
                                                rubric=example["rubric"],
                                                answer_body=example["answer2_body"])
                prompts.append(prompt_a)
                prompts.append(prompt_b)

        answers.append(example["score"])

    print("Prompt built finished! Sampled prompt:")
    if prompts:
        print(prompts[random.randint(0, len(prompts)-1)]+"\n")

    # ==============================================================================
    # 核心修改逻辑：检查 Logit File 是否存在，决定是生成还是读取
    # ==============================================================================
    
    # 这里的变量将在 if/else 中被填充
    predictions = []
    prefix_lens = []
    target_lens = []
    output_ids = []
    pred_scores = []

    if args.logit_file and os.path.exists(args.logit_file):
        print(f"Logit file {args.logit_file} already exists. Loading predictions and skipping VLLM...")
        
        # 1. 读取已保存的 Prediction 和 Score
        with open(args.logit_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["prediction"])
                pred_scores.append(data["score"])
        
        # 确保读取的数量和 Prompt 数量一致
        if len(predictions) != len(prompts):
            print(f"Warning: Prompt count ({len(prompts)}) matches file count ({len(predictions)})? Please check integrity.")
        
        # 2. 重新进行 Tokenize 以获取 output_ids (为了第二步的 Metric 计算)
        # 我们使用 AutoTokenizer，因为它应该与 VLLM 使用的基础模型一致
        print("Re-tokenizing prompts and predictions for evaluation...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        
        # VLLM 默认配置，通常 config 中有 max_model_len
        # 如果获取不到，给一个安全的默认值，或者你需要手动指定
        if hasattr(tokenizer, "model_max_length"):
            MAX_LEN = tokenizer.model_max_length - 512
        else:
            MAX_LEN = 2048 # Fallback

        for prompt, pred in tqdm(zip(prompts, predictions), total=len(prompts), desc="Reconstructing IDs"):
            # 复现 get_multi_answer 中的 Chat Template 逻辑
            if args.apply_chat_template:
                prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = prompt
            
            # 复现 get_multi_answer 中的 Truncation 逻辑
            prompt_ids_list = tokenizer.encode(prompt_text)
            if len(prompt_ids_list) > MAX_LEN:
                prompt_ids_list = prompt_ids_list[-MAX_LEN:]
            
            # Tokenize 预测结果 (注意：不要加 special tokens，因为是接在 prompt 后面的)
            # VLLM 的 output text 通常是纯文本
            pred_ids_list = tokenizer.encode(pred, add_special_tokens=False)

            # 重新组装
            full_ids = prompt_ids_list + pred_ids_list
            
            output_ids.append(full_ids)
            prefix_lens.append(len(prompt_ids_list))
            target_lens.append(len(pred_ids_list))
            
    else:
        print("Logit file not found. Running VLLM inference...")
        # 第一阶段：使用 VLLM 生成所有答案
        predictions, prefix_lens, target_lens, output_ids = get_multi_answer(
            args.model_name_or_path, 
            prompts,
            args.max_new_token, 
            apply_chat_template=args.apply_chat_template
        )

        # 清理 VLLM 显存
        gc.collect()
        torch.cuda.empty_cache()

        # 解析分数
        pred_scores = parse_predictions(predictions, args.model_type, args.data_type, args.prompt_type)

        # 保存结果到 logit_file (修改后的格式: {"score": score, "prediction": prediction})
        if args.logit_file:
            print(f"Saving predictions to {args.logit_file}...")
            with open(args.logit_file, "w", encoding="utf-8") as fout:
                for score, pred in zip(pred_scores, predictions):
                    # 这里保存 score 和 prediction
                    record = {"score": score, "prediction": pred}
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ==============================================================================

    metrics_dicts = calculate_metrics(answers, pred_scores, args.data_type)
    print("**********************************************")
    print(f"Model: {args.model_type}, Data: {args.data_type}, Prompt: {args.prompt_type}")
    print(metrics_dicts)
    print("**********************************************")

    # 初始化结果字典
    results = {"Entropy": []}

    print(f"Loading evaluation model: {args.model_name_or_path}")
    eval_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if eval_tokenizer.pad_token_id is None:
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True
    ).half().to(device)
    model.eval()

    # ----------------------------------------------------------------------
    # 优化后的 Batch Evaluation Loop (Reliability Score)
    # ----------------------------------------------------------------------
    total_samples = len(output_ids)
    batch_size = args.eval_batch_size
    
    print(f"Calculating reliability score with batch size {batch_size}...")
    
    for i in tqdm(range(0, total_samples, batch_size), desc="Scoring Batches"):
        # Slicing
        batch_ids = output_ids[i : i + batch_size]
        batch_prefix = prefix_lens[i : i + batch_size]
        batch_target = target_lens[i : i + batch_size]
        
        batch_results = get_batch_evaluation(
            model, 
            batch_ids, 
            batch_prefix, 
            batch_target,
            pad_token_id=eval_tokenizer.pad_token_id
        )
        
        for res in batch_results:
            results["Entropy"].append(res["entropy"])
            
    # 清理显存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if args.cali_model_name_or_path is not None:
        results["entropy_cali"] = []
        
        print(f"Loading calibration model: {args.cali_model_name_or_path}")
        cali_tokenizer = AutoTokenizer.from_pretrained(args.cali_model_name_or_path)
        if cali_tokenizer.pad_token_id is None:
            cali_tokenizer.pad_token_id = cali_tokenizer.eos_token_id
            
        model = AutoModelForCausalLM.from_pretrained(
            args.cali_model_name_or_path,
            trust_remote_code=True
        ).half().to(device)
        model.eval()

        print(f"Calculating calibration score with batch size {batch_size}...")
        
        for i in tqdm(range(0, total_samples, batch_size), desc="Calibration Batches"):
            batch_ids = output_ids[i : i + batch_size]
            batch_prefix = prefix_lens[i : i + batch_size]
            batch_target = target_lens[i : i + batch_size]
            
            batch_results = get_batch_evaluation(
                model, 
                batch_ids, 
                batch_prefix, 
                batch_target,
                pad_token_id=cali_tokenizer.pad_token_id
            )
            
            for res in batch_results:
                results["entropy_cali"].append(res["entropy"])

    # 将所有结果写入 JSON 文件
    if args.output_file:
        with open(args.output_file, "w") as file_out:
            json.dump(results, file_out, indent=4)
        print(f"All reliability scores have been saved to {args.output_file}.")
    else:
        print("Warning: No output file specified.")