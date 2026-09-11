#!/usr/bin/env python3
"""Verify that this checkout can load every dataset and compute metrics.

Run from the repository root:

    python ds340w/verify_setup.py

Exit status 0 means every check passed. Nothing here needs a GPU, model
weights, an API key, or network access.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_BASE = "fb9e628"  # UnlimitedJudge commit this repo mirrors
UPSTREAM_PATHS = ["src/", "data/", "requirements.txt", "*.sh"]

# Keep the datasets library quiet on the happy path.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PASS, FAIL = "PASS", "FAIL"


def line(status: str, label: str, detail: str = "") -> bool:
    print(f"  [{status}] {label:<44s} {detail}")
    return status == PASS


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ok = True

    # 0. CWD guard. Upstream hard-codes "./data", so the CWD must be the root.
    if Path.cwd().resolve() != REPO_ROOT:
        print("ERROR: run this from the repository root:")
        print(f"       cd {REPO_ROOT}")
        print("       python ds340w/verify_setup.py")
        return 2

    sys.path.insert(0, str(REPO_ROOT))
    import numpy, scipy, sklearn, datasets  # noqa: E401
    from ds340w import data

    try:
        datasets.disable_progress_bars()
        datasets.logging.set_verbosity_error()
    except Exception:  # very old datasets versions
        pass

    # 1. Banner
    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    print("=" * 72)
    print("DS 340W Group 9 - environment verification")
    print("=" * 72)
    print(f"  repo        {REPO_ROOT}")
    print(f"  git         {branch} @ {sha}")
    print(f"  python      {platform.python_version()}  ({sys.executable})")
    print(f"  numpy       {numpy.__version__}")
    print(f"  scipy       {scipy.__version__}")
    print(f"  scikit-learn {sklearn.__version__}")
    print(f"  datasets    {datasets.__version__}")
    print()

    # 2. Repository integrity
    print("Repository checks")
    diff = git("diff", "--stat", UPSTREAM_BASE, "--", *UPSTREAM_PATHS)
    if diff is None:
        ok &= line(FAIL, "upstream files unmodified", f"could not run git diff against {UPSTREAM_BASE}")
    elif diff == "":
        ok &= line(PASS, "upstream files unmodified", f"src/ data/ *.sh requirements.txt == {UPSTREAM_BASE}")
    else:
        ok &= line(FAIL, "upstream files unmodified", diff.splitlines()[-1])
    tracked = git("ls-files")
    n_jsonl = sum(1 for p in (tracked or "").splitlines() if p.endswith(".jsonl"))
    ok &= line(PASS if n_jsonl == 7 else FAIL, "tracked .jsonl data files", f"{n_jsonl} (expected 7)")
    print()

    # 3. Dataset row counts
    keys = [
        "judgelm", "pandalm", "auto-j", "llmbar-natural", "salad-bench",
        "toxic-chat", "halu-eval-qa", "halu-eval-summary", "halu-eval-dialogue",
        "prometheus-ind", "prometheus-ood",
    ]
    print("Dataset loading via upstream src/build_dataset.py")
    loaded: dict[str, list] = {}
    for key in keys:
        expected = data.EXPECTED_ROWS[key]
        try:
            records = data.load(key)
        except Exception as exc:  # report, don't trace
            ok &= line(FAIL, key, f"{type(exc).__name__}: {exc}")
            continue
        loaded[key] = records
        n = len(records)
        ok &= line(PASS if n == expected else FAIL, key, f"n={n:<5d} expected={expected}")
    print()

    # 4. Metric sanity checks with oracle predictions
    print("Metric checks (oracle predictions)")
    if "judgelm" in loaded:
        y = [r["score"] for r in loaded["judgelm"]]
        m = data.calculate_metrics(y, y, "judgelm")
        good = all(abs(m[k] - 1.0) < 1e-12 for k in ("accuracy", "precision", "recall", "f1"))
        ok &= line(PASS if good else FAIL, "judgelm: labels as predictions",
                   " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    if "auto-j" in loaded:
        recs = loaded["auto-j"]
        y = [r["score"] for r in recs]
        m_naive = data.calculate_metrics(y, y, "auto-j")
        m_flip = data.calculate_metrics(y, data.auto_j_oracle_predictions(recs), "auto-j")
        ties = sum(1 for r in recs[: len(recs) // 2] if r["score"] == [1, 1]) / (len(recs) // 2)
        naive_ok = abs(m_naive["consistency"] - ties) < 1e-12 and abs(m_naive["agreement"] - ties) < 1e-12
        flip_ok = m_flip == {"agreement": 1.0, "consistency": 1.0}
        ok &= line(PASS if naive_ok else FAIL, "auto-j: labels as predictions",
                   f"agreement={m_naive['agreement']:.3f} consistency={m_naive['consistency']:.3f} "
                   f"(= tie fraction {ties:.3f}; see docs/auto-j-convention.md)")
        ok &= line(PASS if flip_ok else FAIL, "auto-j: reverse-half flipped oracle",
                   f"agreement={m_flip['agreement']:.3f} consistency={m_flip['consistency']:.3f}")
    print()

    print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
