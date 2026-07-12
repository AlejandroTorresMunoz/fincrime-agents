"""Temporal split invariants: contiguous windows, no overlap, no future in train."""

import pytest

from fincrime_agents.data.split import assign_splits

FRACTIONS = {"train": 0.7, "val": 0.1, "eval": 0.2}


def test_every_case_is_assigned_and_counts_match(cases):
    out = assign_splits(cases, FRACTIONS)
    counts = {s: sum(1 for c in out if c.split == s) for s in ("train", "val", "eval")}
    assert counts == {"train": 70, "val": 10, "eval": 20}


def test_windows_are_temporally_ordered(cases):
    out = assign_splits(cases, FRACTIONS)
    max_train_ts = max(c.alert.ts for c in out if c.split == "train")
    min_val_ts = min(c.alert.ts for c in out if c.split == "val")
    min_eval_ts = min(c.alert.ts for c in out if c.split == "eval")
    assert max_train_ts < min_val_ts < min_eval_ts, "future cases leaked backwards"


def test_input_order_does_not_matter(cases):
    shuffled = list(reversed(cases))
    a = assign_splits(cases, FRACTIONS)
    b = assign_splits(shuffled, FRACTIONS)
    assert [(c.alert.case_id, c.split) for c in a] == [(c.alert.case_id, c.split) for c in b]


def test_input_is_not_mutated(cases):
    assign_splits(cases, FRACTIONS)
    assert all(c.split is None for c in cases)


def test_bad_fractions_are_rejected(cases):
    with pytest.raises(ValueError, match="sum to 1"):
        assign_splits(cases, {"train": 0.5, "val": 0.1, "eval": 0.2})
    with pytest.raises(ValueError, match="missing"):
        assign_splits(cases, {"train": 0.8, "eval": 0.2})
