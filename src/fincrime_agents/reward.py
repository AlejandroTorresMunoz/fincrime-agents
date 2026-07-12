"""The cost/reward matrix — the single source of decision ground truth.

Sparkov's `is_fraud` and the constructed `sanctions_hit` are binary, but the decision
space is ternary (block/review/allow). Rather than inventing a third label, `review`
is an option with a *price*: hedging on a genuinely ambiguous case beats risking a -1,
but the values are calibrated so always answering `review` is never profitable.

This matrix is used in three places and must stay their only source:
  1. SFT filtering  — keep only teacher trajectories whose decision is optimal.
  2. GRPO reward    — the correctness term of the verifiable (RLVR) reward.
  3. Evaluation     — the decision-quality score in the results table.
"""

from __future__ import annotations

from fincrime_agents.schemas import CaseLabel, DecisionAction

# (case_is_positive, action) -> reward. "Positive" = fraud or a true sanctions hit.
REWARD_MATRIX: dict[tuple[bool, DecisionAction], float] = {
    (True, DecisionAction.BLOCK): 1.0,
    (True, DecisionAction.REVIEW): 0.3,  # caught, but costs analyst time
    (True, DecisionAction.ALLOW): -1.0,  # worst case: fraud goes through
    (False, DecisionAction.BLOCK): -1.0,  # legitimate customer blocked
    (False, DecisionAction.REVIEW): -0.2,  # analyst time wasted
    (False, DecisionAction.ALLOW): 1.0,
}


def is_positive(label: CaseLabel) -> bool:
    """A case deserves blocking if it is fraud or a true sanctions match."""
    return label.is_fraud or label.sanctions_hit


def reward(action: DecisionAction, label: CaseLabel) -> float:
    return REWARD_MATRIX[(is_positive(label), action)]


def optimal_action(label: CaseLabel) -> DecisionAction:
    return DecisionAction.BLOCK if is_positive(label) else DecisionAction.ALLOW
