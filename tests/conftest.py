"""Shared test factories: minimal valid Case objects without touching real data."""

import pytest

from fincrime_agents.schemas import Case, CaseAlert, CardProfile, CaseLabel, Transaction


def make_case(i: int, ts: int, is_fraud: bool = False) -> Case:
    """A minimal, schema-valid case for unit tests."""
    tx = Transaction(ts=ts, amount=42.0, category="grocery_pos", merchant="Acme", state="SC")
    return Case(
        alert=CaseAlert(
            case_id=f"case-{i:04d}",
            ts=ts,
            card_id="1234",
            cardholder_name="Jane Doe",
            transaction=tx,
            tcn_score=0.5,
            recent_transactions=[tx],
            profile=CardProfile(
                tx_count=10, median_amount=40.0, top_categories=["grocery_pos"], home_state="SC"
            ),
        ),
        label=CaseLabel(is_fraud=is_fraud),
    )


@pytest.fixture
def cases() -> list[Case]:
    """100 cases with strictly increasing timestamps, every 10th one fraud."""
    return [make_case(i, ts=1_000_000 + i * 3600, is_fraud=i % 10 == 0) for i in range(100)]
