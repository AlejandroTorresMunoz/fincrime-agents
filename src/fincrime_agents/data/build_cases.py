"""Build the final case dataset: fixture -> temporal split -> sanctions augmentation.

    uv run python -m fincrime_agents.data.build_cases

Reads the committed alerts fixture and the downloaded OFAC list, assigns the strict
temporal splits, then applies the sanctions augmentation *within each split* (so the
training and eval windows both get hits and lookalikes — augmenting before splitting
could leave eval without sanctions cases). Deterministic end to end: the same inputs
and seed always produce byte-identical output.

Output: data/processed/cases.json.gz (git-ignored; rebuilt by anyone in seconds).
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from fincrime_agents.config import load_config
from fincrime_agents.data.augment import augment_cases
from fincrime_agents.data.download_sanctions import load_sdn_names
from fincrime_agents.data.split import SPLIT_ORDER, assign_splits
from fincrime_agents.schemas import Case

PROCESSED_PATH = Path("data/processed/cases.json.gz")


def load_fixture(path: str | Path) -> list[Case]:
    """The committed alerts fixture -> validated Case objects."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        fixture = json.load(fh)
    return [Case.model_validate(c) for c in fixture["cases"]]


def load_cases(path: str | Path = PROCESSED_PATH) -> list[Case]:
    """The built dataset (split + augmented), as the graph and training consume it."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [Case.model_validate(c) for c in json.load(fh)]


def build(cfg: dict) -> list[Case]:
    """fixture -> split -> per-split augmentation. Pure; does not touch the filesystem
    beyond reading the two inputs."""
    cases = load_fixture(cfg["paths"]["alerts_fixture"])
    cases = assign_splits(cases, cfg["data"]["split"])

    sdn_names = load_sdn_names(cfg["paths"]["sanctions_list"])
    aug = cfg["data"]["sanctions_augment"]
    out: list[Case] = []
    for i, split in enumerate(SPLIT_ORDER):
        subset = [c for c in cases if c.split == split]
        out.extend(
            augment_cases(
                subset,
                sdn_names,
                fraction=aug["fraction"],
                hit_ratio=aug["hit_ratio"],
                seed=aug["seed"] + i,  # a distinct stream per split, still deterministic
            )
        )
    return out


def main() -> None:
    """CLI entrypoint: build and persist the processed case dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default=PROCESSED_PATH, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cases = build(cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        json.dump([c.model_dump() for c in cases], fh)

    n_sanctions = sum(1 for c in cases if c.label.sdn_entity or c.label.sdn_lookalike_of)
    per_split = {s: sum(1 for c in cases if c.split == s) for s in SPLIT_ORDER}
    print(f"Wrote {len(cases)} cases to {args.out}")
    print(f"  splits: {per_split} | sanctions cases: {n_sanctions}")


if __name__ == "__main__":
    main()
