"""The reward matrix is the single source of decision ground truth — pin its semantics."""

from fincrime_agents.reward import REWARD_MATRIX, is_positive, optimal_action, reward
from fincrime_agents.schemas import CaseLabel, DecisionAction


def test_matrix_covers_every_outcome():
    assert set(REWARD_MATRIX) == {(p, a) for p in (True, False) for a in DecisionAction}


def test_optimal_actions():
    assert optimal_action(CaseLabel(is_fraud=True)) == DecisionAction.BLOCK
    assert optimal_action(CaseLabel(is_fraud=False, sanctions_hit=True)) == DecisionAction.BLOCK
    assert optimal_action(CaseLabel(is_fraud=False)) == DecisionAction.ALLOW


def test_sanctions_hit_is_positive_even_without_fraud():
    assert is_positive(CaseLabel(is_fraud=False, sanctions_hit=True))
    assert not is_positive(CaseLabel(is_fraud=False))


def test_review_is_never_optimal_but_beats_being_wrong():
    fraud, legit = CaseLabel(is_fraud=True), CaseLabel(is_fraud=False)
    for label in (fraud, legit):
        best = reward(optimal_action(label), label)
        hedged = reward(DecisionAction.REVIEW, label)
        assert hedged < best, "hedging must never beat the correct call"
    # ...but on a positive case, review (caught) must beat allow (fraud goes through).
    assert reward(DecisionAction.REVIEW, fraud) > reward(DecisionAction.ALLOW, fraud)


def test_always_review_is_not_profitable_on_a_balanced_mix():
    fraud, legit = CaseLabel(is_fraud=True), CaseLabel(is_fraud=False)
    always_review = reward(DecisionAction.REVIEW, fraud) + reward(DecisionAction.REVIEW, legit)
    honest = reward(DecisionAction.BLOCK, fraud) + reward(DecisionAction.ALLOW, legit)
    assert always_review < honest
