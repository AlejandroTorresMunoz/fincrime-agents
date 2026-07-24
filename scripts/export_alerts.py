"""One-off exporter: build this repo's alerts fixture from the sibling lean-fraud-detection repo.

This is the ONLY step that ever touches the sibling project, and it is run once by the
dataset author on a machine where lean-fraud-detection has been trained (raw CSVs +
processed meta.json + the tcn-raw checkpoint on disk). Its committed outputs make this
repo standalone:

    data/alerts_fixture.json.gz   the cases (alert + ground-truth label), TCN scores frozen in
    data/population_stats.json    fraud base rates by (category, state) — TRAIN split only,
                                  so the agent's tool can never leak test-time labels

It must run inside the SIBLING's environment (torch + lean_fraud), from this repo's root:

    uv run --project ../lean-fraud-detection python scripts/export_alerts.py

Case selection (from the sibling's TEST window only) targets the detector's UNCERTAINTY
band — the triage population. A production cascade auto-blocks slam-dunk scores and
auto-allows obvious negatives; the agent exists for the grey zone, so the fixture must
not be trivially separable by `tcn_score` alone (the detector is near-perfect on
Sparkov, so naive sampling would let the agent learn "threshold the score" and ignore
its tools). Selection:

  - fraud: ALL test frauds are scored; half the quota goes to the hardest ones (lowest
    scores — the detector's misses and near-misses), half to a random sample of the
    rest (ordinary true positives, for realism).
  - legitimate: a large sample (--neg-sample) is scored and the top --max-hard-negatives
    by score are kept — the genuine near-false-positives.

Scoring here batches per card (features built once over each card's full history, then
one window per candidate, inferred in torch batches). The rolling features are causal,
so this is exactly equivalent to the serving path's one-case-at-a-time
`build_feature_window` — just ~50× faster for a bulk export.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAW_HISTORY_COLS = [
    "amt",
    "unix_time",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "category",
    "gender",
    "state",
]


def load_sibling_scorer(sibling: Path):
    """Load the sibling's tcn-raw Scorer with paths made absolute (CWD is this repo)."""
    from lean_fraud.serve.scoring import load_scorer

    cfg = yaml.safe_load((sibling / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["model"]["type"] = "tcn"
    cfg.setdefault("features", {})["engineering"] = "raw"
    cfg.setdefault("mlflow", {})["run_name"] = None  # -> artifacts dir "tcn-raw"
    cfg.setdefault("artifacts", {})["dir"] = str(sibling / "artifacts")
    cfg["dataset"]["processed_dir"] = str(sibling / "data" / "processed")
    return load_scorer(cfg)


def load_raw_transactions(sibling: Path) -> pd.DataFrame:
    """The merged, time-sorted raw table — same ordering rule as the sibling's ETL."""
    frames = [
        pd.read_csv(sibling / "data" / "raw" / name, index_col=0)
        for name in ("fraudTrain.csv", "fraudTest.csv")
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("unix_time", kind="stable").reset_index(drop=True)


def split_boundaries(sibling: Path, n_rows: int) -> tuple[int, int]:
    """Row indices where val and test start, from the sibling's persisted split sizes."""
    meta = json.loads((sibling / "data" / "processed" / "meta.json").read_text(encoding="utf-8"))
    n_train = meta["splits"]["train"]["rows"]
    n_val = meta["splits"]["val"]["rows"]
    assert n_train + n_val < n_rows, "split sizes exceed the raw table"
    return n_train, n_train + n_val


def score_candidates(df: pd.DataFrame, scorer, candidates: list[int]) -> dict[int, float]:
    """TCN probability for each candidate row, batched per card.

    Features are built ONCE over each card's full history (they are causal: row i only
    sees rows <= i), then the (seq_len, n_features) window ending at each candidate is
    sliced out — identical maths to the serving path, minus the per-case rebuild.
    """
    import torch

    from lean_fraud.data.transform.encode import apply_categoricals, apply_scaler
    from lean_fraud.data.transform.features import treat_num_features

    card_rows = df.groupby("cc_num").indices
    by_card: dict[str, list[int]] = defaultdict(list)
    for idx in candidates:
        by_card[df.at[idx, "cc_num"]].append(idx)

    windows: list[np.ndarray] = []
    order: list[int] = []
    for n, (card, idxs) in enumerate(by_card.items(), 1):
        rows = [int(i) for i in card_rows[card]]
        local = {global_i: k for k, global_i in enumerate(rows)}
        sub = df.loc[rows, RAW_HISTORY_COLS].copy().reset_index(drop=True)
        sub["user"] = "card"

        feats, num_cols = treat_num_features(sub, scorer.feats_cfg)
        num_block = feats[num_cols].to_numpy(dtype=np.float32)
        _, code_block = apply_categoricals(feats, scorer.categorical_maps)
        x = np.hstack([num_block, code_block]).astype(np.float32)
        x = apply_scaler(x, scorer.n_numeric, scorer.scaler_mean, scorer.scaler_std)

        for idx in idxs:
            k = local[idx]
            w = x[max(0, k + 1 - scorer.seq_len) : k + 1]
            out = np.zeros((scorer.seq_len, x.shape[1]), dtype=np.float32)
            out[scorer.seq_len - w.shape[0] :] = w
            windows.append(out)
            order.append(idx)
        if n % 100 == 0:
            print(f"  features: {n}/{len(by_card)} cards")

    probs: dict[int, float] = {}
    with torch.no_grad():
        for start in range(0, len(windows), 512):
            batch = torch.from_numpy(np.stack(windows[start : start + 512]))
            for idx, p in zip(order[start : start + 512], torch.sigmoid(scorer.model(batch))):
                probs[idx] = float(p)
    return probs


def population_stats(train_df: pd.DataFrame) -> dict:
    """Fraud base rates the population tool will serve. Train window only (no leakage)."""
    by_cat_state = (
        train_df.groupby(["category", "state"])["is_fraud"].agg(["mean", "size"]).reset_index()
    )
    return {
        "global_fraud_rate": float(train_df["is_fraud"].mean()),
        "by_category": {
            cat: float(rate)
            for cat, rate in train_df.groupby("category")["is_fraud"].mean().items()
        },
        "by_category_state": {
            f"{row.category}|{row.state}": {"fraud_rate": float(row.mean), "n": int(row.size)}
            for row in by_cat_state.itertuples(index=False)
        },
    }


def build_case(df: pd.DataFrame, idx: int, prior_idx: list[int], prob: float, recent: int) -> dict:
    """Assemble one fixture case: the alert (agent-visible) + its ground-truth label."""
    row = df.loc[idx]
    prior = df.loc[prior_idx]

    def tx(r) -> dict:
        return {
            "ts": int(r.unix_time),
            "amount": float(r.amt),
            "category": str(r.category),
            "merchant": str(r.merchant).removeprefix("fraud_"),
            "state": str(r.state),
        }

    top_categories = prior["category"].value_counts().head(3).index.tolist() if len(prior) else []
    profile = {
        "tx_count": int(len(prior)),
        "median_amount": float(prior["amt"].median()) if len(prior) else 0.0,
        "top_categories": [str(c) for c in top_categories],
        "home_state": str(row.state),
    }
    return {
        "alert": {
            "case_id": "",  # assigned after time-sorting all cases
            "ts": int(row.unix_time),
            "card_id": str(row.cc_num),
            "cardholder_name": f"{row['first']} {row['last']}",
            "dob": str(row.dob),
            "nationality": None,
            "transaction": tx(row),
            "tcn_score": round(float(prob), 6),
            "recent_transactions": [tx(r) for r in prior.tail(recent).itertuples(index=False)],
            "profile": profile,
        },
        "label": {
            "is_fraud": bool(row.is_fraud),
            "sanctions_hit": False,
            "sdn_entity": None,
            "sdn_lookalike_of": None,
        },
        "split": None,
    }


def main() -> None:
    """Select the triage population, score it, and write the committed fixture files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sibling", default="../lean-fraud-detection", type=Path)
    parser.add_argument("--out", default="data/alerts_fixture.json.gz", type=Path)
    parser.add_argument("--stats-out", default="data/population_stats.json", type=Path)
    parser.add_argument("--max-fraud", type=int, default=300)
    parser.add_argument("--max-hard-negatives", type=int, default=300)
    parser.add_argument("--neg-sample", type=int, default=50000, help="negatives scored to sample")
    parser.add_argument("--recent", type=int, default=20, help="recent transactions kept per case")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("Loading sibling scorer and raw data...")
    scorer = load_sibling_scorer(args.sibling)
    df = load_raw_transactions(args.sibling)
    n_train, test_start = split_boundaries(args.sibling, len(df))
    card_rows = df.groupby("cc_num").indices  # cc_num -> ordered positional indices

    test_idx = df.index[test_start:]
    fraud_idx = [i for i in test_idx if df.at[i, "is_fraud"] == 1]
    legit_idx = [i for i in test_idx if df.at[i, "is_fraud"] == 0]
    fraud_pick = sorted(fraud_idx)  # score ALL test frauds; stratify after scoring
    legit_pick = sorted(rng.sample(legit_idx, min(args.neg_sample, len(legit_idx))))
    print(f"Scoring {len(fraud_pick)} fraud + {len(legit_pick)} legitimate candidates...")

    scored = score_candidates(df, scorer, fraud_pick + legit_pick)

    # Fraud, stratified by difficulty: half the quota from the LOWEST-scoring frauds
    # (the detector's misses/near-misses — where the agent earns its keep), half sampled
    # from the remaining, ordinarily-caught ones.
    by_score = sorted(fraud_pick, key=lambda i: scored[i])
    n_hard = min(args.max_fraud // 2, len(by_score))
    hard_frauds = by_score[:n_hard]
    easy_pool = by_score[n_hard:]
    easy_frauds = sorted(rng.sample(easy_pool, min(args.max_fraud - n_hard, len(easy_pool))))
    frauds = hard_frauds + easy_frauds

    # Hard negatives: the legitimate transactions the detector was most alarmed by.
    hard_negatives = sorted(legit_pick, key=lambda i: scored[i], reverse=True)
    hard_negatives = hard_negatives[: args.max_hard_negatives]

    selected = sorted(frauds + hard_negatives, key=lambda i: df.at[i, "unix_time"])
    cases = []
    for n, idx in enumerate(selected):
        prior = [int(i) for i in card_rows[df.at[idx, "cc_num"]] if i < idx]
        case = build_case(df, idx, prior, scored[idx], args.recent)
        case["alert"]["case_id"] = f"case-{n:04d}"
        cases.append(case)

    train_df = df.iloc[:n_train]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "provenance": {
            "source": "lean-fraud-detection (Sparkov test window, tcn-raw checkpoint)",
            "detector_threshold": scorer.threshold,
            "selection": "uncertainty-band: hardest frauds + sampled true positives + "
            "top-scoring legitimates",
            "n_fraud_hard": len(hard_frauds),
            "n_fraud_easy": len(easy_frauds),
            "n_hard_negatives": len(hard_negatives),
            "n_fraud_scored": len(fraud_pick),
            "n_legit_scored": len(legit_pick),
            "seed": args.seed,
        },
        "cases": cases,
    }
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        json.dump(fixture, fh)
    args.stats_out.write_text(json.dumps(population_stats(train_df)), encoding="utf-8")

    kb = args.out.stat().st_size / 1024
    n_fraud = sum(c["label"]["is_fraud"] for c in cases)
    print(
        f"Wrote {len(cases)} cases ({n_fraud} fraud / {len(cases) - n_fraud} legit) "
        f"to {args.out} ({kb:.0f} KB) + {args.stats_out}"
    )


if __name__ == "__main__":
    main()
