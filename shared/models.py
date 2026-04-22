"""
Pydantic models and LangGraph state schema for fraud dispute resolution.

These are the system boundaries — all data entering or leaving agents
must conform to these schemas. Validation failures are caught here,
not buried in agent logic.
"""

import operator
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Domain models (Pydantic v2) — used for tool inputs/outputs and LLM parsing
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    """A single card transaction under dispute."""

    id: str
    amount: float = Field(gt=0, description="Transaction amount in USD")
    merchant: str
    merchant_id: str
    date: date
    location: str  # "City, ST"
    device_id: str | None = None


class MerchantRiskProfile(BaseModel):
    """Risk intelligence for a merchant."""

    merchant_id: str
    merchant_name: str
    prior_dispute_count: int = Field(ge=0)
    fraud_rate_pct: float = Field(ge=0.0, le=100.0)
    known_skimmer: bool = False
    flagged_categories: list[str] = Field(default_factory=list)


class CustomerProfile(BaseModel):
    """Customer's historical spend profile."""

    customer_id: str
    name: str
    home_city: str
    recent_transactions: list[dict] = Field(default_factory=list)
    known_devices: list[str] = Field(default_factory=list)
    # Derived — populated by profile agent
    known_locations: list[str] = Field(default_factory=list)
    avg_transaction_amount: float = 0.0


class FraudDecision(BaseModel):
    """
    Structured output the LLM must produce.
    Validated before any state mutation — if fraud_score is outside [0, 1]
    or decision is not a known literal, we catch it here and re-prompt.
    """

    dispute_id: str
    fraud_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability of fraud: 0.0 = certainly legitimate, 1.0 = certainly fraud",
    )
    decision: Literal["auto_credit", "deny", "human_review"]
    resolution_amount: float = Field(
        ge=0.0, description="Amount to credit back (0 if denied)"
    )
    evidence: list[str] = Field(
        description="Key findings that support this decision, one per item"
    )
    reasoning: str = Field(
        description="Brief explanation of how the evidence led to this decision"
    )


# ---------------------------------------------------------------------------
# LangGraph State Schema
# Annotated[list, operator.add] is critical — it lets parallel nodes
# append to `evidence` without stomping each other's writes.
# ---------------------------------------------------------------------------


class DisputeState(TypedDict):
    # Core dispute identity
    dispute_id: str
    customer_id: str

    # Populated by intake node
    transaction: dict  # serialised Transaction

    # Populated by enrich / specialist nodes
    merchant_profile: dict  # serialised MerchantRiskProfile
    customer_profile: dict  # serialised CustomerProfile

    # Accumulated across nodes — Annotated[list, operator.add] makes this
    # safe for parallel fan-out: each branch appends, reducer merges.
    evidence: Annotated[list[str], operator.add]

    # Scoring — updated by score node, never by parallel branches directly
    fraud_score: float

    # Decision lifecycle
    decision: Literal["auto_credit", "deny", "human_review", "pending"]
    resolution_amount: float

    # LLM's scoring reasoning — populated by score/aggregate node in every phase
    model_reasoning: str

    # Human-in-the-loop (Phase 3.2) — analyst notes entered at interrupt only
    human_notes: str
    analyst_approved: bool | None  # None = not yet reviewed

    # Completion
    notification_sent: bool
    error: str | None  # last error message if a node failed
