"""Synthetic sanctions augmentation: give the sanctions agent verifiable cases.

Sparkov cardholders never coincide with OFAC entries, so a fraction of cases is
rewritten so the cardholder name resembles a real SDN individual:

  - **hit** (`sanctions_hit=True`): a light perturbation (typo) of the entry itself,
    with identity attributes that MATCH the entity — the correct call is a match.
  - **lookalike** (`sanctions_hit=False`): a similar-but-different name (first name
    swapped, similarity kept in a lower band) with MISMATCHING attributes — the classic
    false positive naive string matching can't clear; the correct call is no-match.

Ground truth is by construction, not by similarity score. Identity attributes
(dob/nationality) are synthetic but *deterministic per entity* — the screening tool in
the graph derives them with the same `entity_attributes`, so the case evidence and the
tool's answers always agree.
"""

from __future__ import annotations

import hashlib
import random

from rapidfuzz import fuzz

from fincrime_agents.schemas import Case

# Similarity bands (rapidfuzz token_sort_ratio vs. the canonical SDN name).
HIT_MIN_SIMILARITY = 88.0
LOOKALIKE_MIN_SIMILARITY = 70.0
LOOKALIKE_MAX_SIMILARITY = 87.0
_MAX_TRIES = 25

_NATIONALITIES = [
    "Spain",
    "France",
    "Germany",
    "Italy",
    "Portugal",
    "Poland",
    "Romania",
    "Ireland",
    "Lithuania",
    "Greece",
    "Netherlands",
    "Belgium",
    "Austria",
    "Hungary",
    "Croatia",
]

_FIRST_NAMES = [
    "Sergio",
    "Andrei",
    "Miguel",
    "Tomas",
    "Viktor",
    "Rafael",
    "Nikola",
    "Adrian",
    "Emil",
    "Bruno",
    "Ismael",
    "Dario",
    "Marco",
    "Pavel",
    "Hugo",
]


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.lower().encode()).digest()[:8], "big")


def entity_attributes(entity_name: str) -> dict[str, str]:
    """Deterministic synthetic identity attributes for an SDN entity.

    Real SDN birth data lives in free-text remarks and parses unreliably; since the
    cardholders are synthetic anyway, we derive stable attributes from the entity name
    so augmentation and the screening tool can never disagree.
    """
    h = _stable_hash(entity_name)
    year = 1945 + h % 50
    month = 1 + (h >> 8) % 12
    day = 1 + (h >> 16) % 28
    nationality = _NATIONALITIES[(h >> 24) % len(_NATIONALITIES)]
    return {"dob": f"{year:04d}-{month:02d}-{day:02d}", "nationality": nationality}


def mismatched_attributes(entity_name: str) -> dict[str, str]:
    """Attributes near-but-not-equal to the entity's — the lookalike's real identity."""
    attrs = entity_attributes(entity_name)
    h = _stable_hash(entity_name + "|lookalike")
    year = int(attrs["dob"][:4]) + 3 + h % 15
    idx = _NATIONALITIES.index(attrs["nationality"])
    nationality = _NATIONALITIES[(idx + 1 + h % (len(_NATIONALITIES) - 1)) % len(_NATIONALITIES)]
    return {"dob": f"{year:04d}{attrs['dob'][4:]}", "nationality": nationality}


def _typo(name: str, rng: random.Random) -> str:
    """One character-level edit: swap, drop, or double a letter (never the first char)."""
    chars = list(name)
    positions = [i for i in range(1, len(chars) - 1) if chars[i].isalpha()]
    if not positions:
        return name
    i = rng.choice(positions)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap" and i + 1 < len(chars):
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "drop":
        del chars[i]
    else:
        chars.insert(i, chars[i])
    return "".join(chars)


def perturb_toward_hit(canonical: str, rng: random.Random) -> str:
    """A name that IS the entity, with realistic entry noise (>= HIT_MIN_SIMILARITY)."""
    for _ in range(_MAX_TRIES):
        candidate = _typo(canonical, rng)
        if fuzz.token_sort_ratio(candidate, canonical) >= HIT_MIN_SIMILARITY:
            return candidate
    return canonical  # give up perturbing: the exact name is trivially a hit


def perturb_toward_lookalike(canonical: str, rng: random.Random) -> str | None:
    """A different person whose name screens close to the entity (the lookalike band)."""
    tokens = canonical.split()
    for _ in range(_MAX_TRIES):
        candidate_tokens = [rng.choice(_FIRST_NAMES)] + tokens[1:]
        candidate = " ".join(candidate_tokens)
        if rng.random() < 0.5:
            candidate = _typo(candidate, rng)
        score = fuzz.token_sort_ratio(candidate, canonical)
        if LOOKALIKE_MIN_SIMILARITY <= score <= LOOKALIKE_MAX_SIMILARITY:
            return candidate
    return None  # this entity's name doesn't yield a usable lookalike; caller skips it


def augment_cases(
    cases: list[Case],
    sdn_names: list[str],
    fraction: float,
    hit_ratio: float,
    seed: int,
) -> list[Case]:
    """Rewrite a `fraction` of cases into sanctions cases (hits and lookalikes).

    Operates on deep copies; the input list is untouched. Deterministic for a given
    (cases, sdn_names, seed). Call it per split so both training and eval windows get
    their share of sanctions cases.
    """
    rng = random.Random(seed)
    out = [case.model_copy(deep=True) for case in cases]
    n_augment = round(len(out) * fraction)
    if n_augment == 0 or not sdn_names:
        return out

    chosen = rng.sample(range(len(out)), n_augment)
    for j, idx in enumerate(chosen):
        case = out[idx]
        entity = rng.choice(sdn_names)
        if j < round(n_augment * hit_ratio):
            case.alert.cardholder_name = perturb_toward_hit(entity, rng)
            attrs = entity_attributes(entity)
            case.label.sanctions_hit = True
            case.label.sdn_entity = entity
        else:
            name = perturb_toward_lookalike(entity, rng)
            if name is None:
                continue
            case.alert.cardholder_name = name
            attrs = mismatched_attributes(entity)
            case.label.sanctions_hit = False
            case.label.sdn_lookalike_of = entity
        case.alert.dob = attrs["dob"]
        case.alert.nationality = attrs["nationality"]
    return out
