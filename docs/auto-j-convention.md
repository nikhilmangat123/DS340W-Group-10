# Auto-J forward/reverse convention in upstream code

**Status: resolved, with evidence. Pinned by `ds340w/tests/test_autoj_convention.py`.**

## The question

`src/build_dataset.py::calculate_metrics` handles `data_type == "auto-j"` by
splitting the predictions into a forward half and a reverse half, negating
the reverse half, and then counting

- `consistency`: pairs where the (negated) reverse verdict equals the forward verdict, and
- `agreement`: consistent pairs whose forward verdict also equals the label.

Feeding the dataset's own labels back in as predictions, which should behave
like a perfect oracle, returned

```
{'agreement': 0.26795977011494254, 'consistency': 0.26795977011494254}
```

instead of 1.0. Two hypotheses were on the table: (a) the reverse-half labels
are stored pre-flipped in the JSONL, or (b) there is an indexing quirk.

## What the data and code actually do

**1. The JSONL contains one ordering only.** `data/auto-j/testdata_pairwise.jsonl`
has **1,392 rows**, not 2,784. Each row has the keys
`scenario, label, prompt, response 1, response 2`. Checking every row against
every other row, no row is the response-swapped copy of another row. So there
is no "reverse half" in the file to be pre-flipped or otherwise.

Label distribution in the file: `0` → 520 rows, `1` → 499 rows, `2` → 373 rows.
Per the Auto-J README (GAIR-NLP/auto-j), `label` is the human annotation where
`0` means response 1 is preferred, `1` means response 2, and `2` means tie.
The file is 58 scenarios × 24 pairs.

**2. `build_dataset("auto-j")` synthesizes the reverse half.** The loader
(`src/build_dataset.py`, `elif data_type == "auto-j":`) first maps each row to
upstream's unified format, `score_mapping = {"0": [1, 0], "1": [0, 1], "2": [1, 1]}`,
then does:

```python
reve_dataset = []
for example in dataset:
    rev_example = copy.deepcopy(example)
    temp_body = rev_example["answer1_body"]
    rev_example["answer1_body"] = rev_example["answer2_body"]
    rev_example["answer2_body"] = temp_body
    reve_dataset.append(rev_example)
dataset.extend(reve_dataset)
```

The answer bodies are swapped. **The `score` is not.** Verified on the loaded
list: for all 1,392 values of *i*, record *i*+1392 has the same `question_body`
as record *i*, its `answer1_body`/`answer2_body` swapped, and an identical
`score`. So in the 2,784-row list, the reverse-half `score` is expressed in the
*forward* ordering's frame and is wrong for the ordering actually shown.

**3. `calculate_metrics` never reads the reverse half of `y_true`.** It uses
`y_true[:n/2]` only. It negates `y_pred[n/2:]` because a position-invariant
judge that sees the swapped pair should output the opposite verdict, and
negating maps that back into the forward frame. Replacing the reverse half of
`y_true` with garbage (`[9, 9]`) changes nothing.

## Why the oracle scored 0.268

With labels as predictions, the forward-half verdict *w* and the reverse-half
verdict are identical (because the score was copied, not flipped). After
negation the comparison is *w == −w*, which holds only when *w* = 0, i.e. a tie.
Ties are 373 of 1,392 forward pairs:

```
373 / 1392 = 0.26795977…
```

which is exactly the value observed for both `consistency` and `agreement`.

Supplying the correct oracle, forward scores as stored plus reverse scores
reversed (`[1,0]` ↔ `[0,1]`, `[1,1]` unchanged), gives

```
{'agreement': 1.0, 'consistency': 1.0}
```

So neither hypothesis (a) nor (b) is right. The labels are not pre-flipped, and
there is no indexing bug in `calculate_metrics`. The function is written for
*model outputs*, which naturally flip under swapping, and the "oracle" test fed
it labels that had been copied unflipped by the loader.

## Consequences for this project

- **Never treat rows `[1392:]` of `build_dataset("auto-j")` as independently
  labelled records.** Their `score` describes the other ordering.
- **A correct oracle for `calculate_metrics(..., "auto-j")` must flip the
  reverse half.** `ds340w.data.auto_j_oracle_predictions()` does this.
- **`ds340w.data.auto_j_pairs()`** returns `(forward, reverse, label)` triples
  where each record's `score` is correct for its own ordering and `label` is the
  forward-frame human preference (`"answer1"`, `"answer2"`, `"tie"`). This is
  the structure the planned consistent/flipped × correct/wrong decomposition
  needs.
- **Upstream's `consistency` is a same-verdict metric.** A judge that always
  prefers whichever answer is shown first scores `consistency = 1.0` and
  `agreement = 520/1392 = 0.3736` on this data. That is the motivating case for
  the project and is pinned in
  `test_position_consistent_but_wrong_judge_maxes_consistency`.

## Minor observation

Rows 1272 and 1273 of the JSONL share a prompt and each has `response 1 ==
response 2` (both labelled tie). They are legitimate tie items, not a hidden
reverse pair, and do not affect anything above.

## Reproduce

```bash
python ds340w/tests/test_autoj_convention.py
python ds340w/verify_setup.py      # prints both the 0.268 and the 1.0 oracle lines
```

Verified 2026-09-10 against upstream commit `fb9e628`.
