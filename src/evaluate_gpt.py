import json
import argparse
import random
import time
import json
import openai
import os
import re
import requests
import multiprocessing
from functools import partial
import timeout_decorator

from evaluate_judge import build_dataset, calculate_metrics
from build_prompt_gpt import parse_score_gpt, create_prompt_gpt


def build_params_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--prompt-type", type=str, choices=("vanilla", "cot"), default=None)
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench",
                 "halu-eval-summary", "halu-eval-qa", "halu-eval-dialogue", "salad-bench", "toxic-chat",
                 "llmbar-neighbor", "llmbar-natural", "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
        default=None,
    )
    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--max-new-token", type=int, default=None, help="The maximum number of new tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="The temperature for sampling.")
    parser.add_argument("--logit-file", type=str, default=None)
    parser.add_argument("--pool-number", type=int, default=32)
    parser.add_argument("--multi-process", type=str, default="False")
    parser.add_argument("--rewrite-output", type=str, default="False")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for saving results.")
    args = parser.parse_args()
    return args

@timeout_decorator.timeout(30, use_signals=True)
def request_gpt(prompt, model, temperature, max_new_tokens):
    url = "https://yunwu.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer PUT-YOUR-KEY-HERE",
    }
    max_tries = 5
    res = ''
    response = None
    sys_info = {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."}
    for i in range(max_tries):
        try:
            messages = [{"role": "user", "content": prompt}]
            data = {"model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_new_tokens}
            response = requests.post(
                url, headers=headers, data=json.dumps(data))
            response_json = response.json()
            if 'choices' in response_json:
                res = response_json['choices'][0]['message']['content'].strip()
                break
            else:
                time.sleep(2)
                continue
        except Exception as e:
            print("Exception! The response is " + str(response))
            time.sleep(5)
            continue
    if i == max_tries - 1:
        print("Max tries exceeded! Fallback to empty response.")
    return res

def gpt_scoring(prompt, model, temperature, max_new_tokens):
    prediction = request_gpt(prompt, model, temperature=temperature, max_new_tokens=max_new_tokens)

    # 还原回原始的counter逻辑
    counter.value += 1
    print(f"gpt_scoring {counter.value} finished.")

    return prediction


def init(c):
    global counter
    counter = c


if __name__ == "__main__":
    args = build_params_gpt()
    random.seed(42)

    if "prometheus" in args.data_type:
        args.prompt_type = "cot"
    
    if args.max_new_token is None:
        if args.prompt_type == "cot":
            args.max_new_token = 1024
        else:
            args.max_new_token = 16

    dataset = build_dataset(args.data_type, args.data_path)
    instruction = create_prompt_gpt(args.data_type, args.prompt_type)

    prompts = []
    answers = []
    for example in dataset:
        if args.data_type in ["prometheus-ind", "prometheus-ood", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa", "toxic-chat"]:
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer_body=example["answer_body"])
            prompts.append(prompt)
        else:
            example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
            if args.prompt_type == "icl":
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            demonstrations=example["demonstrations"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
            else:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])      
            prompts.append(prompt)

        answers.append(example["score"])

    print("Prompt built finished! Sampled prompt:")
    if len(prompts) > 0:
        print(prompts[random.randint(0, len(prompts)-1)]+"\n")

    if args.logit_file is None:
        args.logit_file = f"./outputs/{args.data_type}-{args.model_name}-{args.prompt_type}.jsonl"

    # -----------------------------------------------------------
    # Resume Logic (断点续传检测)
    # -----------------------------------------------------------
    processed_count = 0
    existing_pred_scores = []
    
    # 如果不要求重写，且文件存在，则读取已有的进度
    if args.rewrite_output == "False" and os.path.exists(args.logit_file):
        print(f"File {args.logit_file} exists. Checking processed lines...")
        with open(args.logit_file, "r") as f:
            lines = f.readlines()
            processed_count = len(lines)
            for line in lines:
                data = json.loads(line)
                existing_pred_scores.append(data["score"])
        print(f"Resuming from index {processed_count}. Total: {len(prompts)}.")
    elif args.rewrite_output == "True" and os.path.exists(args.logit_file):
        os.remove(args.logit_file)

    # 如果已经全部跑完了，直接跳过推理
    if processed_count >= len(prompts):
        print("All data already processed. Calculating metrics directly...")
        metrics_dicts = calculate_metrics(answers, existing_pred_scores, args.data_type)
        print("**********************************************")
        print(f"Model: {args.model_name}, Data: {args.data_type}")
        print(metrics_dicts)
        print("**********************************************")
        exit()

    # 只处理剩下的 Prompt
    remaining_prompts = prompts[processed_count:]

    # -----------------------------------------------------------
    # Counter Init (保持原始逻辑，但设置初始值为已完成数量)
    # -----------------------------------------------------------
    manager = multiprocessing.Manager()
    # counter 初始化为 processed_count，这样日志打印的数字是连续的
    counter = manager.Value("counter", processed_count)
    
    # -----------------------------------------------------------
    # Multiprocessing Pool Init
    # -----------------------------------------------------------
    if args.multi_process != "False":
        pool = multiprocessing.Pool(processes=args.pool_number, initializer=init, initargs=(counter,))
    else:
        # 单进程模式下，手动调用init以确保全局counter被设置，防止NameError
        init(counter)

    # -----------------------------------------------------------
    # Batch Processing & Writing
    # -----------------------------------------------------------
    new_pred_scores = []
    batch_size = args.batch_size
    total_remaining = len(remaining_prompts)

    print(f"Start processing {total_remaining} items in batches of {batch_size}...")

    for i in range(0, total_remaining, batch_size):
        batch_prompts = remaining_prompts[i : i + batch_size]
        
        # 批量处理
        if args.multi_process == "False":
            predictions = [gpt_scoring(sample, model=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_token)
                           for sample in batch_prompts]
        else:
            pool_fn = partial(gpt_scoring, model=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_token)
            predictions = pool.map(pool_fn, batch_prompts)

        # 解析分数
        current_scores = [parse_score_gpt(p, data_type=args.data_type, prompt_type=args.prompt_type) for p in predictions]
        new_pred_scores.extend(current_scores)

        # 写入文件 (Append 模式)
        with open(args.logit_file, "a") as f:
            for idx, prediction in enumerate(predictions):
                json_line = {
                    "score": current_scores[idx],
                    "prediction": prediction,
                }
                f.write(json.dumps(json_line) + "\n")
        
        print(f"Batch saved. Progress: {i + len(batch_prompts)}/{total_remaining} (relative to resume point)")

    if args.multi_process != "False":
        pool.close()
        pool.join()

    # -----------------------------------------------------------
    # Final Metrics
    # -----------------------------------------------------------
    final_pred_scores = existing_pred_scores + new_pred_scores
    metrics_dicts = calculate_metrics(answers, final_pred_scores, args.data_type)
    
    print("**********************************************")
    print(f"Model: {args.model_name}, Data: {args.data_type}")
    print(metrics_dicts)
    print("**********************************************")