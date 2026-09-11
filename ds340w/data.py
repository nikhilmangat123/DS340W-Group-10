"""Thin, CWD-independent wrapper around upstream ``src/build_dataset.py``.

Why this exists
---------------
Upstream's ``build_dataset(data_type, data_path="./data")`` has three traits
that make it awkward to call from anywhere but an interactive shell at the
repo root:

1. Several branches ignore ``data_path`` and hard-code ``"./data"`` or
   ``"data/..."``, so it only works when the current working directory is the
   repository root.
2. The bare keys ``"halu-eval"`` and ``"prometheus"`` fall through every
   ``elif`` and raise ``UnboundLocalError`` instead of a useful message.
3. For ``"auto-j"`` it appends a response-swapped copy of every record whose
   ``score`` is *not* flipped (see ``docs/auto-j-convention.md``). A caller
   that treats the second half as independently labelled records gets wrong
   labels.

This module does not patch upstream. It imports upstream unchanged, wraps the
call in a ``chdir`` to the repo root, validates the key, and (for auto-j)
exposes the forward/reverse structure explicitly.

Reproducibility note
--------------------
Upstream's ``pandalm`` and ``halu-eval-*`` branches call the unseeded global
``random`` module while assigning labels, so two consecutive upstream loads
can disagree on some labels. ``load()`` therefore seeds ``random`` before
calling upstream (``seed=0`` by default). Pass ``seed=None`` to reproduce raw
upstream behaviour. Row counts are unaffected either way.
"""
from __future__ import annotations

import contextlib
import os
import random
import sys
from pathlib import Path
from typing import Iterator, NamedTuple

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
UPSTREAM_SRC: Path = REPO_ROOT / "src"
AUTO_J_RAW: Path = REPO_ROOT / "data" / "auto-j" / "testdata_pairwise.jsonl"

# Every key that upstream build_dataset() handles without error.
VALID_KEYS: tuple[str, ...] = (
    "judgelm",
    "pandalm",
    "auto-j",
    "llmbar-natural",
    "llmbar-neighbor",
    "llmbar-gptinst",
    "llmbar-gptout",
    "llmbar-manual",
    "salad-bench",
    "toxic-chat",
    "halu-eval-qa",
    "halu-eval-summary",
    "halu-eval-dialogue",
    "prometheus-ind",
    "prometheus-ood",
)

# Keys a reader might reasonably try that upstream does NOT handle.
_BROKEN_KEYS: dict[str, str] = {
    "halu-eval": "use one of halu-eval-qa, halu-eval-summary, halu-eval-dialogue",
    "prometheus": "use prometheus-ind or prometheus-ood",
    "llmbar": "use llmbar-natural, llmbar-neighbor, llmbar-gptinst, llmbar-gptout or llmbar-manual",
}

# Row counts after upstream's own filtering/truncation, verified 2026-09-10.
EXPECTED_ROWS: dict[str, int] = {
    "judgelm": 4849,
    "pandalm": 999,
    "auto-j": 2784,
    "llmbar-natural": 100,
    "llmbar-neighbor": 134,
    "llmbar-gptinst": 92,
    "llmbar-gptout": 47,
    "llmbar-manual": 46,
    "salad-bench": 1920,
    "toxic-chat": 1000,
    "halu-eval-qa": 1000,
    "halu-eval-summary": 1000,
    "halu-eval-dialogue": 1000,
    "prometheus-ind": 1000,
    "prometheus-ood": 1000,
}


class InvalidDatasetKey(ValueError):
    pass


def validate_key(name: str) -> str:
    if name in VALID_KEYS:
        return name
    hint = _BROKEN_KEYS.get(name)
    msg = f"Unknown dataset key {name!r}."
    if hint:
        msg += f" Upstream raises UnboundLocalError for this bare name; {hint}."
    msg += " Valid keys: " + ", ".join(VALID_KEYS)
    raise InvalidDatasetKey(msg)


@contextlib.contextmanager
def in_repo_root() -> Iterator[Path]:
    """Temporarily chdir to the repository root (upstream hard-codes ./data)."""
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield REPO_ROOT
    finally:
        os.chdir(previous)


def upstream():
    """Import and return upstream's ``build_dataset`` module, unmodified."""
    src = str(UPSTREAM_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import build_dataset  # noqa: WPS433  (upstream module, imported lazily)

    return build_dataset


def load(name: str, seed: int | None = 0) -> list[dict]:
    """Load a dataset through upstream ``build_dataset`` from any CWD.

    Returns upstream's records unchanged. For ``"auto-j"`` remember that rows
    ``[n/2:]`` are response-swapped copies of rows ``[:n/2]`` whose ``score``
    was copied without flipping; use :func:`auto_j_pairs` instead if you need
    per-record labels that are correct in their own ordering.
    """
    validate_key(name)
    if seed is not None:
        random.seed(seed)
    with in_repo_root():
        return upstream().build_dataset(name)


def calculate_metrics(y_true: list, y_pred: list, name: str) -> dict:
    """Pass-through to upstream ``calculate_metrics`` with key validation."""
    validate_key(name)
    return upstream().calculate_metrics(y_true, y_pred, name)


# --------------------------------------------------------------------------
# auto-j: explicit forward/reverse pairs
# --------------------------------------------------------------------------

class AutoJPair(NamedTuple):
    """One Auto-J comparison presented in both orders.

    ``forward``  upstream record i           (answer1 = response 1, answer2 = response 2)
    ``reverse``  upstream record i + n/2, with its ``score`` corrected so that
                 it describes the swapped ordering (i.e. ``forward["score"][::-1]``).
    ``label``    the human preference in the FORWARD ordering:
                 ``"answer1"``, ``"answer2"`` or ``"tie"``.
    """

    forward: dict
    reverse: dict
    label: str


_SCORE_TO_LABEL = {(1, 0): "answer1", (0, 1): "answer2", (1, 1): "tie"}


def score_to_label(score: list[int]) -> str:
    return _SCORE_TO_LABEL[tuple(score)]


def auto_j_pairs(seed: int | None = 0) -> list[AutoJPair]:
    """Return the 1,392 Auto-J comparisons as (forward, reverse, label) triples.

    Upstream stores the pairing only implicitly (first half / second half of a
    flat 2,784-row list) and leaves the reverse record's ``score`` unflipped.
    Here each record's ``score`` is correct for its own ordering, and the pair
    structure is explicit. Records are copies; upstream's list is not mutated.
    """
    records = load("auto-j", seed=seed)
    half = len(records) // 2
    pairs: list[AutoJPair] = []
    for i in range(half):
        fwd = dict(records[i])
        rev = dict(records[i + half])
        assert fwd["answer1_body"] == rev["answer2_body"], f"row {i}: not a swap"
        assert fwd["answer2_body"] == rev["answer1_body"], f"row {i}: not a swap"
        rev["score"] = list(reversed(fwd["score"]))
        pairs.append(AutoJPair(fwd, rev, score_to_label(fwd["score"])))
    return pairs


def auto_j_oracle_predictions(records: list[dict]) -> list[list[int]]:
    """Predictions a perfect, position-invariant judge would give for upstream's
    2,784-row auto-j list, in the layout ``calculate_metrics`` expects:
    forward-half scores as stored, reverse-half scores flipped.
    """
    half = len(records) // 2
    forward = [list(r["score"]) for r in records[:half]]
    return forward + [s[::-1] for s in forward]
