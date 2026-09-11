# Decision log

Running record of project decisions, newest at the bottom. Each entry states
what was decided, why, and what it commits us to.

---

## D1 — Select Huang et al. (2025) as the first parent paper
**Date:** 2026-09-10

**Decision.** The first parent paper is Huang, Bu, Zhou, Qu, Liu, Yang, Xu,
Zhao (2025), *An Empirical Study of LLM-as-a-Judge for LLM Evaluation:
Fine-tuned Judge Model is not a General Substitute for GPT-4*, Findings of
ACL 2025, pp. 5880–5895. https://aclanthology.org/2025.findings-acl.306/

**Against the four assignment criteria.**

| criterion | evidence |
|---|---|
| 1. Peer-reviewed, 2024 or later | Published in Findings of the Association for Computational Linguistics: ACL 2025 (Vienna, July 27–Aug 1, 2025). Not a preprint. |
| 2. Detailed methodology, results, conclusions | 16-page paper evaluating four fine-tuned judge models (JudgeLM, PandaLM, Prometheus, Auto-J) against GPT-4 across eight test sets on generalisability, fairness and adaptability, with a stated conclusion that fine-tuned judges behave as task-specific classifiers. |
| 3. Public datasets | All eight evaluation sets are public and are committed inside the code repository under `data/`; sources listed in `NOTICE.md` and `dataset-summary.md`. |
| 4. Public code | https://github.com/HuihuiChyan/UnlimitedJudge, which describes itself as the paper's official repository. The dataset-loading and metric code runs on CPU with no API key (verified: `ds340w/verify_setup.py`). |

**Why this one over the alternatives.** See `parent-papers.md`. In short: it
is the only one of the candidates that is (a) peer-reviewed at a first-tier
NLP venue, (b) ships code *and* data in one place, and (c) already implements
a forward/reverse pairwise protocol (`consistency` on Auto-J) that our
correctness decomposition can be built directly on top of.

**Consequences.** The project's data substrate is upstream's `data/` and its
`build_dataset` / `calculate_metrics` functions. Their quirks (D4, D5, D6)
become our constraints.

---

## D2 — Preserve upstream git history instead of squashing or re-initialising
**Date:** 2026-09-10

**Decision.** This repository was created by a full (non-shallow) clone of
UnlimitedJudge, retaining all 209 upstream commits, with our work layered on
top. Upstream is kept as the `upstream` remote.

**Why.**
- *Provenance.* Anyone can see exactly which upstream commit (`fb9e628`) we
  started from and can verify with one command that we have not altered it:
  `git diff --stat fb9e628 -- src/ data/ requirements.txt '*.sh'` must print
  nothing. `verify_setup.py` runs this check.
- *Honesty of NOTICE.md.* Upstream has no LICENSE. The defensible position is
  "we redistribute the authors' work unchanged, with attribution, for
  coursework"; a squashed or edited copy would weaken that.
- *A concrete trap.* Upstream's `.gitignore` contains `*.jsonl`, yet seven
  `.jsonl` data files (~80 MB) are tracked because they were committed before
  the rule was added. Any `git init` + `git add .` silently drops them.
  Preserving history keeps them; the gate `git ls-files | grep -c '\.jsonl$'`
  must print 7.

**Consequences.** We never rewrite history or force-push. Our own changes are
confined to `ds340w/`, `docs/`, `NOTICE.md`, `.gitignore` and a header at the
top of `README.md` (D7).

---

## D3 — Bypass the root `requirements.txt`
**Date:** 2026-09-10

**Decision.** We install `ds340w/requirements.txt` (numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0, datasets 5.0.1) and never the root file.

**Why.** The root file pins torch 2.1.2, vllm 0.4.0 and flash-attn 2.0.4. Those
exist to run 7B–13B judge checkpoints on a GPU in `evaluate_judge.py`,
`cal_reliability.py`, `evaluate_finetuned.py` and `finetune.py`, none of which
we can or need to run (no GPU; `evaluate_gpt.py` additionally needs a paid
OpenAI key). The code path we use, `build_dataset` and `calculate_metrics`,
imports only numpy, scipy, scikit-learn and `datasets`. Building the 2024 GPU
stack on a 2026 machine is slow and fragile and buys nothing.

**Consequences.** Anything in the project that requires running a judge model
will need its own, separately documented environment. Our analysis stays
CPU-only and reproducible from four pinned packages.

---

## D4 — Wrap upstream code; never patch it in place
**Date:** 2026-09-10

**Decision.** Awkward behaviours in `src/build_dataset.py` are handled in
`ds340w/data.py`, not by editing upstream. Specifically: hard-coded `./data`
paths (we `chdir` around the call), bare keys that raise `UnboundLocalError`
(we validate against a whitelist), and the Auto-J reverse-half scores (D6).

**Why.** Editing upstream would break the byte-identity guarantee in D2 and
force us to argue that our modified copy still measures what the paper
measured. Wrapping keeps the paper's code as the fixed reference point.

**Consequences.** All project code imports `ds340w.data`, never
`src/build_dataset.py` directly.

---

## D5 — Seed upstream's unseeded randomness in the wrapper
**Date:** 2026-09-10

**Decision.** `ds340w.data.load(name, seed=0)` calls `random.seed(seed)` before
invoking upstream. `seed=None` reproduces raw upstream behaviour.

**Why.** Upstream's `pandalm` loader (via a duplicated condition in its
majority-vote code) and `halu-eval-*` loaders (a coin flip per row) use the
unseeded global `random` module while assigning labels. Measured on
back-to-back raw loads: 21/999 pandalm labels and 502/1000 halu-eval-qa labels
changed. Any reliability metric computed on unstable labels is not
reproducible. Details in `dataset-summary.md`.

**Consequences.** Our numbers on pandalm and halu-eval are reproducible but
correspond to one specific realisation of upstream's randomness. When we
compare against figures in the paper we must note that the paper's own
realisation is unknown.

---

## D6 — Auto-J reverse-half labels are handled explicitly
**Date:** 2026-09-10

**Decision.** Code that needs Auto-J pairs uses `ds340w.data.auto_j_pairs()`,
which returns `(forward, reverse, label)` triples with each record's `score`
correct for its own ordering. Oracle predictions for upstream's
`calculate_metrics(..., "auto-j")` come from
`ds340w.data.auto_j_oracle_predictions()`.

**Why.** The JSONL holds 1,392 rows in a single ordering. Upstream's loader
appends a response-swapped copy of each with the `score` copied unflipped, and
`calculate_metrics` compensates by negating the reverse half of the
*predictions* while ignoring the reverse half of the labels. Feeding labels
back as predictions therefore scores 0.268 (the tie fraction), not 1.0. Full
evidence in `auto-j-convention.md`; pinned by
`ds340w/tests/test_autoj_convention.py`.

**Consequences.** The consistent/flipped × correct/wrong decomposition is built
on `auto_j_pairs()`, not on the flat 2,784-row list.

---

## D7 — `README.md` is the one upstream file we edit
**Date:** 2026-09-10

**Decision.** A DS 340W header is prepended to `README.md`; the upstream text
follows unchanged. No other upstream file is touched.

**Why.** Upstream's first sentence, "This is the official repository for
paper ...", is false when read as a statement about *this* repository. A
reader must see who we are, what we mirrored, and that we are not the
upstream authors before reaching that sentence. A separate file would not
achieve that because GitHub renders `README.md` on the landing page.

**Consequences.** `NOTICE.md` records this as the single edited upstream
file. The byte-identity check in D2 deliberately excludes `README.md`.
