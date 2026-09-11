"""Regression tests pinning the Auto-J forward/reverse convention.

These encode the findings in docs/auto-j-convention.md. If upstream's data
file or loader ever changes, or if someone "fixes" the wrapper in a way that
breaks the convention, these fail.

Run with either:
    python -m pytest ds340w/tests
    python ds340w/tests/test_autoj_convention.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ds340w import data  # noqa: E402

RAW_ROWS = 1392
TIES_RAW = 373          # label == 2 in the JSONL
RESPONSE1_RAW = 520     # label == 0
RESPONSE2_RAW = 499     # label == 1


def _raw():
    with open(data.AUTO_J_RAW) as fh:
        return [json.loads(l) for l in fh]


def test_raw_file_is_single_ordering():
    rows = _raw()
    assert len(rows) == RAW_ROWS
    assert set(rows[0]) == {"scenario", "label", "prompt", "response 1", "response 2"}
    labels = [r["label"] for r in rows]
    assert labels.count(0) == RESPONSE1_RAW
    assert labels.count(1) == RESPONSE2_RAW
    assert labels.count(2) == TIES_RAW
    # No row is the response-swapped copy of another row -> the file itself
    # contains one ordering only.
    seen = {(r["prompt"], r["response 1"], r["response 2"]) for r in rows}
    swapped = sum((r["prompt"], r["response 2"], r["response 1"]) in seen
                  for r in rows if r["response 1"] != r["response 2"])
    assert swapped == 0


def test_upstream_loader_appends_unflipped_swapped_copies():
    recs = data.load("auto-j")
    half = len(recs) // 2
    assert len(recs) == 2 * RAW_ROWS
    for i in range(half):
        f, r = recs[i], recs[i + half]
        assert f["question_body"] == r["question_body"]
        assert f["answer1_body"] == r["answer2_body"]
        assert f["answer2_body"] == r["answer1_body"]
        # The trap: score is copied, not flipped.
        assert f["score"] == r["score"]


def test_labels_fed_back_as_predictions_score_only_on_ties():
    recs = data.load("auto-j")
    y = [r["score"] for r in recs]
    m = data.calculate_metrics(y, y, "auto-j")
    expected = TIES_RAW / RAW_ROWS
    assert abs(m["consistency"] - expected) < 1e-12
    assert abs(m["agreement"] - expected) < 1e-12


def test_flipped_reverse_half_is_a_perfect_oracle():
    recs = data.load("auto-j")
    y = [r["score"] for r in recs]
    m = data.calculate_metrics(y, data.auto_j_oracle_predictions(recs), "auto-j")
    assert m == {"agreement": 1.0, "consistency": 1.0}


def test_calculate_metrics_ignores_reverse_half_of_y_true():
    recs = data.load("auto-j")
    half = len(recs) // 2
    y_true_garbage = [r["score"] for r in recs[:half]] + [[9, 9]] * half
    m = data.calculate_metrics(y_true_garbage, data.auto_j_oracle_predictions(recs), "auto-j")
    assert m == {"agreement": 1.0, "consistency": 1.0}


def test_position_consistent_but_wrong_judge_maxes_consistency():
    """A judge that always prefers whichever answer is shown first is 100%
    consistent under upstream's metric and right only as often as response 1
    happens to be the human-preferred one."""
    recs = data.load("auto-j")
    half = len(recs) // 2
    y = [r["score"] for r in recs]
    always_first = [[1, 0]] * half + [[0, 1]] * half
    m = data.calculate_metrics(y, always_first, "auto-j")
    assert m["consistency"] == 1.0
    assert abs(m["agreement"] - RESPONSE1_RAW / RAW_ROWS) < 1e-12


def test_auto_j_pairs_have_frame_correct_scores():
    pairs = data.auto_j_pairs()
    assert len(pairs) == RAW_ROWS
    labels = [p.label for p in pairs]
    assert labels.count("answer1") == RESPONSE1_RAW
    assert labels.count("answer2") == RESPONSE2_RAW
    assert labels.count("tie") == TIES_RAW
    for p in pairs:
        assert p.reverse["score"] == p.forward["score"][::-1]
        assert p.forward["answer1_body"] == p.reverse["answer2_body"]


def test_invalid_keys_raise_helpful_error():
    for bad in ("halu-eval", "prometheus", "llmbar", "nonsense"):
        try:
            data.load(bad)
        except data.InvalidDatasetKey as exc:
            assert "Valid keys" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should have been rejected")


if __name__ == "__main__":
    import inspect
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
