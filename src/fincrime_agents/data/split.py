"""Strict temporal split of cases — the agent-level anti-leakage discipline.

Cases are ordered by the flagged transaction's timestamp and cut into contiguous
windows: the oldest go to the agent's train/val (SFT trajectories + GRPO rollouts),
the most recent are the untouchable final eval. Same invariant as the sibling repo's
transaction split: nothing from the future leaks into training, and no case is ever
both trained on and evaluated.
"""

from __future__ import annotations

from fincrime_agents.schemas import Case

SPLIT_ORDER = ["train", "val", "eval"]


def assign_splits(cases: list[Case], fractions: dict[str, float]) -> list[Case]:
    """Return cases sorted by time with `split` set per contiguous window.

    `fractions` maps split name -> fraction; it must cover SPLIT_ORDER and sum to ~1.
    """
    missing = [name for name in SPLIT_ORDER if name not in fractions]
    if missing:
        raise ValueError(f"fractions missing splits: {missing}")
    total = sum(fractions[name] for name in SPLIT_ORDER)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")

    ordered = sorted((case.model_copy(deep=True) for case in cases), key=lambda c: c.alert.ts)
    n = len(ordered)
    n_train = int(n * fractions["train"])
    n_val = int(n * fractions["val"])
    for i, case in enumerate(ordered):
        if i < n_train:
            case.split = "train"
        elif i < n_train + n_val:
            case.split = "val"
        else:
            case.split = "eval"
    return ordered
