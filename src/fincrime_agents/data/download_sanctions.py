"""Download the OFAC SDN list (public domain, U.S. Treasury) to a local CSV.

The raw file is deliberately NOT committed — run this once during setup:

    uv run python -m fincrime_agents.data.download_sanctions

The classic sdn.csv has no header row and uses "-0-" as its null marker. We keep the
columns the project needs (entity number, name, type, program, remarks) and only
individuals (the cardholder-screening use case).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fincrime_agents.config import load_config

# Column layout of the classic OFAC sdn.csv (fixed order, no header row).
SDN_COLUMNS = [
    "ent_num",
    "name",
    "type",
    "program",
    "title",
    "call_sign",
    "vess_type",
    "tonnage",
    "grt",
    "vess_flag",
    "vess_owner",
    "remarks",
]
KEEP_COLUMNS = ["ent_num", "name", "type", "program", "remarks"]
NULL_MARKER = "-0-"


def canonical_name(sdn_name: str) -> str:
    """Normalize an SDN entry ("LAST, First Middle") to display order ("First Middle Last")."""
    if "," in sdn_name:
        last, _, rest = sdn_name.partition(",")
        return f"{rest.strip()} {last.strip()}".strip().title()
    return sdn_name.strip().title()


def parse_sdn_csv(source: str | Path) -> pd.DataFrame:
    """Read a raw sdn.csv (path or URL) into a tidy individuals-only frame."""
    df = pd.read_csv(source, names=SDN_COLUMNS, header=None, dtype=str, keep_default_na=False)
    # Fields arrive with stray whitespace and "-0- " (trailing space) as the null marker.
    df = df.apply(lambda col: col.str.strip()).replace(NULL_MARKER, "")
    df = df[df["type"].str.lower() == "individual"][KEEP_COLUMNS].reset_index(drop=True)
    df["canonical_name"] = df["name"].map(canonical_name)
    return df


def load_sdn_names(path: str | Path) -> list[str]:
    """Canonical individual names from an already-downloaded ofac_sdn.csv."""
    return pd.read_csv(path, dtype=str)["canonical_name"].dropna().tolist()


def main() -> None:
    """CLI entrypoint: download, tidy, and persist the SDN individuals list."""
    # Trust the OS certificate store (like curl/browsers do): Python's bundled CAs
    # reject corporate/AV TLS-interception chains, which are common on Windows.
    import truststore

    truststore.inject_into_ssl()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--url", default=None, help="Override the SDN CSV URL from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    url = args.url or cfg["data"]["sdn_url"]
    dest = Path(cfg["paths"]["sanctions_list"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    df = parse_sdn_csv(url)
    df.to_csv(dest, index=False)
    print(f"Saved {len(df)} SDN individuals to {dest}")


if __name__ == "__main__":
    main()
