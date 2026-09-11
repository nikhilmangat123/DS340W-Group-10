# Dataset summary

All eight evaluation sets used by the parent paper ship inside this repository
under `data/`, exactly as the upstream authors committed them. Nothing is
downloaded at run time. Record structures and counts below were obtained by
opening the files and by calling upstream's `build_dataset()` on 2026-09-10
(upstream commit `fb9e628`); they are not copied from papers.

## Counts as actually loaded

`build_dataset(key)` filters, truncates or expands several files. The table
gives both the raw file size and what the loader returns.

| key | file(s) | raw rows | loaded rows | how the loader gets there |
|---|---|---|---|---|
| `judgelm` | `judgelm/judgelm_val_5k.jsonl` + `judgelm_val_5k_gpt4.jsonl` | 5,000 + 5,000 | **4,849** | joins the two files by position; drops 151 rows whose GPT-4 score is `[-1, -1]` |
| `pandalm` | `pandalm/testset-v1.json` | 999 | **999** | 1:1; label = majority of three annotators (see caveat) |
| `auto-j` | `auto-j/testdata_pairwise.jsonl` | 1,392 | **2,784** | appends a response-swapped copy of every row ([details](auto-j-convention.md)) |
| `llmbar-natural` | `llmbar/natural/dataset.json` | 100 | **100** | 1:1 |
| `llmbar-neighbor` | `llmbar/neighbor/dataset.json` | 134 | 134 | 1:1 |
| `llmbar-gptinst` | `llmbar/gptinst/dataset.json` | 92 | 92 | 1:1 |
| `llmbar-gptout` | `llmbar/gptout/dataset.json` | 47 | 47 | 1:1 |
| `llmbar-manual` | `llmbar/manual/dataset.json` | 46 | 46 | 1:1 |
| `salad-bench` | `salad-bench/mcq_set.json` | 3,840 | **1,920** | keeps even-indexed rows; builds one safe/unsafe answer pair per row with `random.seed(42)` |
| `toxic-chat` | `toxic-chat/data_0124_toxic-chat_annotation_test.csv` | 5,083 | **1,000** | shuffles with `random.seed(42)`, keeps the first 1,000 |
| `halu-eval-qa` | `halu-eval/qa.jsonl` | 10,000 | **1,000** | first 1,000 rows; one output per row chosen by coin flip |
| `halu-eval-summary` | `halu-eval/summary.jsonl` | 10,000 | **1,000** | same |
| `halu-eval-dialogue` | `halu-eval/dialogue.jsonl` | 10,000 | **1,000** | same |
| `prometheus-ind` | `prometheus/feedback_collection_test.json` | 1,000 | **1,000** | 1:1 |
| `prometheus-ood` | `prometheus/feedback_collection_ood_test.json` | 1,000 | **1,000** | 1:1 |

Bold rows are the eleven keys checked by `ds340w/verify_setup.py`.
`halu-eval/general_data.jsonl` (4,507 rows) is in the repository but no
upstream loader reads it.

## Upstream's unified record format

Every loader rewrites its source into one of two shapes.

Pairwise (judgelm, pandalm, auto-j, llmbar-*, salad-bench):

```python
{"question_body": str, "answer1_body": str, "answer2_body": str, "score": [s1, s2]}
```

`score` is a two-element list. For judgelm it holds GPT-4's two 1–10 scores.
For every other pairwise set it is already a win indicator:
`[1, 0]` = answer 1 preferred, `[0, 1]` = answer 2 preferred, `[1, 1]` = tie.
`calculate_metrics` converts both forms to a win list (`1`, `-1`, `0`) by
comparing `s1` and `s2`.

Pointwise (prometheus-*, halu-eval-*, toxic-chat):

```python
{"question_body": str, "answer_body": str, "rubric": str, "score": int}
```

`score` is 1–5 for prometheus and `0`/`1` for halu-eval and toxic-chat.
judgelm records additionally keep upstream's original metadata fields
(`question_id`, `answer1_model_id`, `answer2_model_id`, ...).

## Determinism caveat

Two loaders assign labels with the **unseeded** global `random` module, so two
consecutive raw `build_dataset` calls can return different labels for the
same rows:

- `pandalm`: the majority-vote code reads
  `if line["annotator1"] == line["annotator2"] or line["annotator1"] == line["annotator2"]`.
  The second clause repeats the first (presumably `annotator3` was intended),
  so the 49 rows where annotators 1 and 3 agree against annotator 2 fall
  through to `random.choice` over the three votes. In one pair of back-to-back
  loads, 21 of 999 labels differed. There are no rows where all three
  annotators disagree.
- `halu-eval-*`: each of the 1,000 rows gets `right_*` (score 1) or
  `hallucinated_*` (score 0) by `random.random() >= 0.5`. In one pair of
  back-to-back loads, 502 of 1,000 labels differed.

`toxic-chat` and `salad-bench` call `random.seed(42)` themselves and are
stable. judgelm, auto-j, llmbar and prometheus involve no randomness.
`ds340w.data.load` seeds `random` (default `seed=0`) before every upstream
call so that our results are reproducible; row counts are unaffected.

## Per-dataset detail

### 1. JudgeLM validation set — `data/judgelm/`
- **Source:** https://huggingface.co/datasets/BAAI/JudgeLM-100K (Zhu, Wang, Wang; BAAI, 2023)
- **Raw record, `judgelm_val_5k.jsonl`:** `question_id`, `question_body`, `answer1_body`, `answer2_body`, `answer1_model_id`, `answer2_model_id`, `answer1_metadata`, `answer2_metadata`, and a `score` field that is a pair of dicts of generation statistics (logprobs, ROUGE, BLEU), not a judgment.
- **Raw record, `judgelm_val_5k_gpt4.jsonl`:** `review_id`, `question_id`, `reviewer_id`, `text` (GPT-4's free-text review beginning with two numbers, e.g. `"8 7\nAssistant 1 provided..."`), `score` (those two numbers as floats, e.g. `[8.0, 7.0]`; `[-1, -1]` when unparseable).
- **Label semantics:** the reference verdict is GPT-4's score pair; higher score wins, equal scores tie. After loading: 2,327 answer-1 wins, 2,303 answer-2 wins, 219 ties.
- **Upstream metric:** accuracy, macro precision/recall/F1 on the three-way win list.

### 2. PandaLM test set — `data/pandalm/`
- **Source:** https://github.com/WeOpenML/PandaLM/blob/main/data/testset-v1.json (Wang et al., ICLR 2024)
- **Raw record:** `idx`, `motivation_app`, `cmp_key` (which two models produced the responses, e.g. `bloom-7b_llama-7b`), `instruction`, `input`, `response1`, `response2`, `annotator1`, `annotator2`, `annotator3`.
- **Label semantics:** each annotator gives `0` = similar quality (tie), `1` = response 1 better, `2` = response 2 better (PandaLM README). Upstream maps to `[1,1]`, `[1,0]`, `[0,1]` after the majority vote described above. Question text is `input + "\n" + instruction` when `input` is non-empty.
- **Upstream metric:** accuracy, macro precision/recall/F1.

### 3. Auto-J pairwise test set — `data/auto-j/`
- **Source:** https://github.com/GAIR-NLP/auto-j/blob/main/data/test/testdata_pairwise.jsonl (Li et al., GAIR-NLP, 2023)
- **Raw record:** `scenario` (one of 58, 24 rows each), `label`, `prompt`, `response 1`, `response 2`.
- **Label semantics:** human annotation, `0` = response 1 preferred (520 rows), `1` = response 2 preferred (499), `2` = tie (373). Upstream maps to `[1,0]`, `[0,1]`, `[1,1]`, then appends a swapped copy of every record **without flipping the score**. See [auto-j-convention.md](auto-j-convention.md).
- **Upstream metric:** `agreement` and `consistency` over forward/reverse pairs.

### 4. Feedback Bench (Prometheus) — `data/prometheus/`
- **Source:** https://github.com/kaistAI/prometheus/tree/main/evaluation/benchmark/data (Kim et al., KAIST AI, 2023)
- **Raw record:** `idx`, `instruction` (a single prompt string that embeds the task description, the instruction to evaluate, the response to evaluate, a reference answer and a 1–5 score rubric under `###` headings), `gpt4_score`, `gpt4_feedback`.
- **Label semantics:** `gpt4_score` is GPT-4's 1–5 rubric score; both files hold exactly 200 items per score level. Upstream regex-extracts `question_body`, `answer_body` and `rubric` out of `instruction`. `-ind` is the in-distribution test split, `-ood` the out-of-distribution one.
- **Upstream metric:** Pearson, Kendall τ, Spearman.

### 5. LLMBar — `data/llmbar/`
- **Source:** https://github.com/princeton-nlp/LLMBar/tree/main/Dataset/LLMBar (Zeng et al., ICLR 2024)
- **Raw record:** `input`, `output_1`, `output_2`, `label`.
- **Label semantics:** `label ∈ {1, 2}` indicating which output is *objectively* better (no ties; LLMBar README). Upstream maps `1 → [1,0]`, `2 → [0,1]`. Subsets: Natural (100) and the adversarial Neighbor (134), GPTInst (92), GPTOut (47), Manual (46); 419 total, matching the LLMBar README.
- **Upstream metric:** accuracy, macro precision/recall/F1.

### 6. HaluEval — `data/halu-eval/`
- **Source:** https://github.com/RUCAIBox/HaluEval/tree/main/data (Li et al., RUCAIBox, 2023)
- **Raw records:** `qa.jsonl`: `knowledge`, `question`, `right_answer`, `hallucinated_answer`. `summary.jsonl`: `document`, `right_summary`, `hallucinated_summary`. `dialogue.jsonl`: `knowledge`, `dialogue_history`, `right_response`, `hallucinated_response`. `general_data.jsonl` (unused): `ID`, `user_query`, `chatgpt_response`, `hallucination` (`yes`/`no`), `hallucination_spans`.
- **Label semantics after loading:** `score = 1` when the shown output is the right one, `0` when it is the hallucinated one; the choice is the unseeded coin flip noted above. A fixed hallucination-detection rubric string is attached.
- **Upstream metric:** sweeps ten thresholds over the judge's numeric output and reports the best accuracy.

### 7. ToxicChat (0124 release) — `data/toxic-chat/`
- **Source:** https://huggingface.co/datasets/lmsys/toxic-chat (Lin et al., LMSYS, 2023)
- **Raw record (CSV):** `conv_id`, `user_input`, `model_output`, `human_annotation`, `toxicity` (0/1; 362 of 5,083 rows are 1), `jailbreaking` (0/1; 91 rows are 1), `openai_moderation` (per-category scores).
- **Label semantics after loading:** `score = 1 - toxicity`, so `1` = non-toxic/safe, `0` = toxic. A fixed safety rubric string is attached. Note upstream's own comment on the file name: it loads the *test* split even though the `datasets` library labels it `train`.
- **Upstream metric:** threshold sweep, best macro F1.

### 8. SALAD-Bench MCQ set — `data/salad-bench/`
- **Source:** https://huggingface.co/datasets/OpenSafetyLab/Salad-Data (Li et al., OpenSafetyLab, 2024)
- **Raw record:** `mcq` (full multiple-choice prompt), `baseq` (the underlying question), `choices` (a single string `"A. ...; B. ...; C. ..."`), `gt` (letters of the safe choices), `1-category`, `2-category`, `3-category` (safety taxonomy).
- **Label semantics after loading:** upstream parses the three choices, picks one at random as answer 1; if it is in `gt` the pair is `score = [1, 0]` and answer 2 is a random unsafe choice, otherwise `score = [0, 1]` and answer 2 is a random safe choice. So `[1,0]` means answer 1 is the safe response.
- **Upstream metric:** accuracy, macro precision/recall/F1.

## Not present locally

Upstream's README also lists MT-Bench human judgments
(https://huggingface.co/datasets/lmsys/mt_bench_human_judgments). There is no
`data/mt-bench/` directory and no `build_dataset` branch for it in the mirrored
commit, so it is not part of this repository.
