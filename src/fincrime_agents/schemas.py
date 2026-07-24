"""Core data contracts shared by the dataset builders, the graph, and training.

The key design split: `CaseAlert` is everything the agent is allowed to see, while
`CaseLabel` is the verifiable ground truth (Sparkov's `is_fraud` + the constructed
`sanctions_hit`). Keeping them in separate models makes it structurally impossible to
leak a label into a prompt or a tool response by accident.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """One raw transaction, as the tools expose it (a subset of Sparkov's columns)."""

    ts: int = Field(description="Unix timestamp (seconds)")
    amount: float
    category: str
    merchant: str
    state: str


class CardProfile(BaseModel):
    """Behavioural baseline of a card, precomputed at export time from its full history."""

    tx_count: int
    median_amount: float
    top_categories: list[str]
    home_state: str


class CaseAlert(BaseModel):
    """What the investigator graph sees: the flagged transaction plus its context.

    `dob` / `nationality` exist so the sanctions specialist has attributes to
    disambiguate against a list entry; they are synthetic (Sparkov cardholders are
    synthetic people) but internally consistent with the screening tool.
    """

    case_id: str
    ts: int = Field(description="Unix timestamp of the flagged transaction")
    card_id: str
    cardholder_name: str
    dob: str | None = None
    nationality: str | None = None
    transaction: Transaction
    tcn_score: float = Field(ge=0.0, le=1.0, description="The cheap detector's fraud probability")
    recent_transactions: list[Transaction] = Field(default_factory=list)
    profile: CardProfile


class CaseLabel(BaseModel):
    """Verifiable ground truth. Never shown to the agent; used only to filter SFT
    trajectories, compute the GRPO reward, and score evaluations."""

    is_fraud: bool
    sanctions_hit: bool = False
    sdn_entity: str | None = Field(
        default=None, description="The real SDN entry this case was perturbed toward (hits only)"
    )
    sdn_lookalike_of: str | None = Field(
        default=None, description="The SDN entry a hard-negative name resembles (lookalikes only)"
    )


class Case(BaseModel):
    """A dataset row: the alert, its ground truth, and its temporal split assignment."""

    alert: CaseAlert
    label: CaseLabel
    split: str | None = Field(default=None, description="train | val | eval")


class DecisionAction(str, Enum):
    BLOCK = "block"
    REVIEW = "review"
    ALLOW = "allow"


class TriageDecision(BaseModel):
    """The graph's final structured output for a case."""

    action: DecisionAction
    rationale: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SpecialistFinding(BaseModel):
    """One specialist agent's contribution, kept as auditable structured evidence."""

    agent: str
    summary: str
    evidence: list[str] = Field(default_factory=list)
