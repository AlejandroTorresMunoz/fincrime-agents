"""Sanctions augmentation invariants: verifiable labels, similarity bands, determinism."""

import random

from rapidfuzz import fuzz

from fincrime_agents.data.augment import (
    HIT_MIN_SIMILARITY,
    LOOKALIKE_MAX_SIMILARITY,
    LOOKALIKE_MIN_SIMILARITY,
    augment_cases,
    entity_attributes,
    mismatched_attributes,
    perturb_toward_hit,
    perturb_toward_lookalike,
)

SDN_NAMES = [
    "Ivan Petrov",
    "Manuel Garcia Lopez",
    "Omar Al-Rashid",
    "Viktor Kovalenko",
    "Hassan El-Masri",
]


def test_augmentation_is_deterministic(cases):
    a = augment_cases(cases, SDN_NAMES, fraction=0.2, hit_ratio=0.5, seed=13)
    b = augment_cases(cases, SDN_NAMES, fraction=0.2, hit_ratio=0.5, seed=13)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


def test_input_is_not_mutated(cases):
    augment_cases(cases, SDN_NAMES, fraction=0.2, hit_ratio=0.5, seed=13)
    assert all(c.label.sdn_entity is None and c.label.sdn_lookalike_of is None for c in cases)


def test_fraction_and_hit_ratio_are_respected(cases):
    out = augment_cases(cases, SDN_NAMES, fraction=0.2, hit_ratio=0.5, seed=13)
    hits = [c for c in out if c.label.sanctions_hit]
    lookalikes = [c for c in out if c.label.sdn_lookalike_of]
    assert len(hits) == 10  # 100 cases * 0.2 fraction * 0.5 hit_ratio
    assert len(hits) + len(lookalikes) <= 20  # lookalike generation may skip an entity
    assert len(lookalikes) >= 8


def test_hits_match_their_entity_attributes(cases):
    out = augment_cases(cases, SDN_NAMES, fraction=0.2, hit_ratio=0.5, seed=13)
    for case in (c for c in out if c.label.sanctions_hit):
        attrs = entity_attributes(case.label.sdn_entity)
        assert case.alert.dob == attrs["dob"]
        assert case.alert.nationality == attrs["nationality"]
        score = fuzz.token_sort_ratio(case.alert.cardholder_name, case.label.sdn_entity)
        assert score >= HIT_MIN_SIMILARITY


def test_lookalikes_mismatch_their_entity_attributes(cases):
    out = augment_cases(cases, SDN_NAMES, fraction=0.2, hit_ratio=0.5, seed=13)
    for case in (c for c in out if c.label.sdn_lookalike_of):
        entity = case.label.sdn_lookalike_of
        attrs = entity_attributes(entity)
        assert (case.alert.dob, case.alert.nationality) != (attrs["dob"], attrs["nationality"])
        score = fuzz.token_sort_ratio(case.alert.cardholder_name, entity)
        assert LOOKALIKE_MIN_SIMILARITY <= score <= LOOKALIKE_MAX_SIMILARITY
        assert not case.label.sanctions_hit


def test_entity_attributes_are_stable():
    assert entity_attributes("Ivan Petrov") == entity_attributes("ivan petrov")
    assert mismatched_attributes("Ivan Petrov") != entity_attributes("Ivan Petrov")


def test_perturbations_stay_in_their_bands():
    rng = random.Random(0)
    for name in SDN_NAMES:
        hit = perturb_toward_hit(name, rng)
        assert fuzz.token_sort_ratio(hit, name) >= HIT_MIN_SIMILARITY
        lookalike = perturb_toward_lookalike(name, rng)
        if lookalike is not None:
            score = fuzz.token_sort_ratio(lookalike, name)
            assert LOOKALIKE_MIN_SIMILARITY <= score <= LOOKALIKE_MAX_SIMILARITY
