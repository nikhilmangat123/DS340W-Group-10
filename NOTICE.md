# NOTICE

This repository is a mirror of third-party code and data, assembled as the
"parent paper" base for a DS 340W (Applied Data Sciences, Penn State) course
project on reliability metrics for LLM-as-a-Judge. It is kept for coursework
and reproducibility purposes. All rights in the code and data remain with their
respective authors.

## Upstream code

The code in `src/` and the shell scripts in the repository root are copied,
with full git history, from:

- **UnlimitedJudge** — https://github.com/HuihuiChyan/UnlimitedJudge
  — the official code release of the paper cited below. Its first author is
  **Hui Huang** (Faculty of Computing, Harbin Institute of Technology);
  "HuihuiChyan" is only the GitHub account name. Co-authors are affiliated
  with Harbin Institute of Technology, Beijing Institute of Technology and
  Baidu Inc., per the paper's first page.

It is the official repository for:

> Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang,
> Bing Xu, Tiejun Zhao. 2025. *An Empirical Study of LLM-as-a-Judge for LLM
> Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4.*
> In Findings of the Association for Computational Linguistics: ACL 2025,
> pages 5880–5895. https://aclanthology.org/2025.findings-acl.306/

The upstream repository does **not** include a LICENSE file. No explicit
redistribution or reuse grant has been stated by its authors, so this code
should not be described as open source. It is reproduced here for
non-commercial academic study only. If you are an upstream author and object
to this mirror, please open an issue and it will be taken down.

Changes made in this mirror relative to upstream (base commit `fb9e628`):

- `README.md`: a DS 340W project header is prepended above the upstream
  README text, which follows unchanged. This is the only upstream file that
  has been edited, and the header exists so that upstream's opening sentence
  ("This is the official repository for paper ...") is not read as a claim
  about this repository.
- `.gitignore`: entries appended for model weights, run outputs and
  virtual environments.
- Added, not present upstream: this `NOTICE.md`; `ds340w/` (Group 9's own
  code: environment spec, verification script, data-access wrapper, tests);
  `docs/` (Group 9's documentation).

No upstream source files were modified. `src/`, `data/`, `requirements.txt`
and the root `*.sh` scripts are byte-identical to upstream commit `fb9e628`,
which can be checked with
`git diff --stat fb9e628 -- src/ data/ requirements.txt '*.sh'` (empty output).

## Third-party datasets under `data/`

The evaluation sets below were collected by the upstream authors and are
redistributed here unchanged from the upstream repository. Each belongs to its
original creators; consult the linked source for its license and terms of use.

| Directory | Dataset | Source | Contents in this repo |
|---|---|---|---|
| `data/judgelm/` | **JudgeLM-100K** validation set (Zhu, Wang, Wang; BAAI, 2023) | https://huggingface.co/datasets/BAAI/JudgeLM-100K | `judgelm_val_5k.jsonl` (instruction + two model answers) and `judgelm_val_5k_gpt4.jsonl` (GPT-4 pairwise scores used as reference labels). |
| `data/pandalm/` | **PandaLM** human-annotated test set (Wang et al., 2023) | https://github.com/WeOpenML/PandaLM/blob/main/data/testset-v1.json | `testset-v1.json`: pairwise comparisons with human preference labels (win / lose / tie). |
| `data/auto-j/` | **Auto-J** pairwise test set (Li et al., GAIR-NLP, 2023) | https://github.com/GAIR-NLP/auto-j/blob/main/data/test/testdata_pairwise.jsonl | `testdata_pairwise.jsonl`: 1,392 human-labeled response pairs across 58 scenarios (evaluated in both orderings upstream). |
| `data/prometheus/` | **Feedback Bench** from Prometheus (Kim et al., KAIST AI, 2023) | https://github.com/kaistAI/prometheus/tree/main/evaluation/benchmark/data | `feedback_collection_test.json` (in-distribution) and `feedback_collection_ood_test.json` (out-of-distribution): single-response 1–5 rubric-based scoring with GPT-4 reference scores. |
| `data/llmbar/` | **LLMBar** (Zeng et al., Princeton NLP, 2023) | https://github.com/princeton-nlp/LLMBar/tree/main/Dataset/LLMBar | `natural/` plus the adversarial subsets `neighbor/`, `gptinst/`, `gptout/`, `manual/`, each a `dataset.json` of instruction-following pairs with objective preference labels. |
| `data/halu-eval/` | **HaluEval** (Li et al., RUCAIBox, 2023) | https://github.com/RUCAIBox/HaluEval/tree/main/data | `qa.jsonl`, `summary.jsonl`, `dialogue.jsonl`, `general_data.jsonl`: hallucination-detection samples pairing a correct and a hallucinated LLM output. |
| `data/toxic-chat/` | **ToxicChat** (Lin et al., LMSYS, 2023), 0124 release | https://huggingface.co/datasets/lmsys/toxic-chat | `data_0124_toxic-chat_annotation_test.csv`: real user prompts with human toxicity / jailbreak annotations. |
| `data/salad-bench/` | **SALAD-Bench** (Li et al., OpenSafetyLab, 2024) | https://huggingface.co/datasets/OpenSafetyLab/Salad-Data | `mcq_set.json`: multiple-choice safety questions used to test judges on safety-related evaluation. |

Please cite the original dataset papers and the upstream paper above in any
work that uses this repository.
