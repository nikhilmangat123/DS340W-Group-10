# `ds340w/` — Group 10 project code

Everything in this directory is written by DS 340W Group 10. It wraps the
upstream UnlimitedJudge code in `../src/` and never modifies it.

## Setup (CPU only, no API key, no model weights)

From the repository root:

```bash
python3 -m venv .venv             # Python 3.12.12 verified; any 3.10+ should work
source .venv/bin/activate
pip install -r ds340w/requirements.txt
```

Do **not** run `pip install -r requirements.txt` at the repository root. That
file is upstream's GPU stack (torch 2.1.2, vllm 0.4.0, flash-attn 2.0.4) for
running 7B–13B judge models and is not needed for anything in `ds340w/`. See
`ds340w/requirements.txt` for the pinned versions we verified.

## Verify the setup

```bash
python ds340w/verify_setup.py
```

This loads all eleven working dataset keys through upstream's
`build_dataset`, checks the row counts against the expected values, checks
that the upstream files are byte-identical to the mirrored commit, and runs
oracle sanity checks on `calculate_metrics`. It exits non-zero if anything
fails and prints no stack traces on the happy path.

Regression tests for the Auto-J ordering convention
(`../docs/auto-j-convention.md`):

```bash
python ds340w/tests/test_autoj_convention.py     # plain python
python -m pytest ds340w/tests                     # if pytest is installed
```

## Loading data from your own code

```python
from ds340w import data

records = data.load("auto-j")                 # upstream records, any CWD
pairs   = data.auto_j_pairs()                 # (forward, reverse, label) triples
m       = data.calculate_metrics(y_true, y_pred, "judgelm")
```

`data.load(name)` resolves the repository root from its own location, changes
directory into it for the duration of the upstream call, restores the previous
directory, and validates `name` against the whitelist below.

### Valid dataset keys

| key | task | rows | notes |
|---|---|---|---|
| `judgelm` | pairwise, GPT-4 reference | 4849 | 151 rows with unparseable GPT-4 scores dropped upstream |
| `pandalm` | pairwise, human | 999 | label assignment uses unseeded `random` upstream; see below |
| `auto-j` | pairwise, human | 2784 | 1392 pairs × 2 orderings; second half synthesized by upstream |
| `llmbar-natural` | pairwise, objective | 100 | |
| `llmbar-neighbor` | pairwise, adversarial | 134 | |
| `llmbar-gptinst` | pairwise, adversarial | 92 | |
| `llmbar-gptout` | pairwise, adversarial | 47 | |
| `llmbar-manual` | pairwise, adversarial | 46 | |
| `salad-bench` | pairwise, safety | 1920 | every second MCQ row, seeded |
| `toxic-chat` | pointwise, toxicity | 1000 | seeded shuffle, first 1000 of 5083 |
| `halu-eval-qa` | pointwise, hallucination | 1000 | first 1000 of 10000; coin flip per row upstream |
| `halu-eval-summary` | pointwise, hallucination | 1000 | same |
| `halu-eval-dialogue` | pointwise, hallucination | 1000 | same |
| `prometheus-ind` | pointwise 1–5, GPT-4 reference | 1000 | |
| `prometheus-ood` | pointwise 1–5, GPT-4 reference | 1000 | |

The bare names `halu-eval`, `prometheus` and `llmbar` are **not** valid: upstream
raises `UnboundLocalError` for them. The wrapper raises `InvalidDatasetKey`
with the list above instead.

### Working-directory constraint

Upstream's `build_dataset` hard-codes `./data` in several branches, so calling
it directly only works when the current directory is the repository root.
`ds340w.data.load` handles this for you; `verify_setup.py` refuses to run from
anywhere else and tells you where to `cd`.

### Randomness

Upstream's `pandalm` and `halu-eval-*` loaders call the global `random` module
without seeding while assigning labels, so two raw upstream loads can disagree.
`data.load` seeds `random` (default `seed=0`) before each call so results are
reproducible. Pass `seed=None` to get raw upstream behaviour.

## What we do not run

`src/evaluate_judge.py`, `src/cal_reliability.py`, `src/evaluate_finetuned.py`
and `src/finetune.py` import `vllm`/`torch` and need 7B–13B checkpoints on a
GPU. `src/evaluate_gpt.py` needs a paid OpenAI key. None are exercised by
anything in `ds340w/`.
