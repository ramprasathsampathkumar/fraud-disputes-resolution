"""
Phase 3.1 — Persistence & Long-term Memory

Concepts demonstrated:
  - Short-term memory: SqliteSaver checkpoints every node transition.
    Re-running with the same thread_id resumes from the last checkpoint —
    disputes survive process restarts, analyst laptop closes, etc.
  - Long-term memory: ChromaDB stores cross-dispute learnings.
    After each resolution, merchant and customer patterns are embedded.
    Future disputes query this memory before scoring — memory changes outcomes.

Graph structure (extends Phase 2.2 with memory nodes):

    intake → memory_lookup → fan_out ──┬──→ merchant_check ──┐
                                       ├──→ customer_check ──┼──→ aggregate → [decide] → outcome → store_outcome → notify
                                       └──→ velocity_check ──┘

Key additions vs Phase 2.2:
  - memory_lookup: queries ChromaDB BEFORE parallel checks — enriches state
    with prior knowledge (travel patterns, confirmed fraud merchants)
  - store_outcome: runs AFTER notify — embeds dispute resolution into ChromaDB
    so future disputes on the same merchant/customer benefit from this learning

The "aha" demo (DISP-001):
  Cold run:  No memory → Miami location flagged → score 0.91 → AUTO-CREDIT
  Seed:      "CUST-4821 travels to Miami every April" stored in ChromaDB
  Warm run:  Memory retrieved → location no longer anomalous → score ~0.55 → HUMAN REVIEW

This shows memory meaningfully changing outcomes — not just logging decisions.

Run:
    # Cold run (clears memory first)
    uv run python -m phase3_1_persistence.graph --dispute DISP-001

    # Seed long-term memory with CUST-4821 travel pattern
    uv run python -m phase3_1_persistence.graph --seed-memory

    # Warm run (memory already seeded)
    uv run python -m phase3_1_persistence.graph --dispute DISP-001 --warm

    # Show all stored memories
    uv run python -m phase3_1_persistence.graph --show-memory

    # Demonstrate short-term resumability (re-run same thread_id)
    uv run python -m phase3_1_persistence.graph --dispute DISP-001 --warm --resume
"""

import argparse
import asyncio
import logging
import operator
import os
import time
import warnings
from typing import Annotated, Literal

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from typing_extensions import TypedDict

load_dotenv()

warnings.filterwarnings("ignore", message="create_react_agent has been moved", category=DeprecationWarning)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from shared.mock_data import DISPUTES
from shared.model_factory import current_config, get_investigator
from shared.tools import (
    check_merchant_fraud_history,
    check_velocity_pattern,
    get_customer_spend_history,
    get_transaction,
)

console = Console()

_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if _ls_key and not _ls_key.startswith(("ls__", "lsv2_")):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "dispute_memory"


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB client
# ═══════════════════════════════════════════════════════════════════════════════

def get_chroma_collection() -> chromadb.Collection:
    """
    Returns the persistent ChromaDB collection.
    PersistentClient stores embeddings to disk at CHROMA_PATH — survives
    process restarts, unlike the in-memory client used in dev tutorials.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# State schema
# Extends Phase 2.2's ParallelState with memory_context.
# ═══════════════════════════════════════════════════════════════════════════════

class PersistenceState(TypedDict):
    # ── Core dispute fields (same as Phase 2.2) ──────────────────────────────
    dispute_id: str
    customer_id: str
    transaction: dict
    merchant_profile: dict
    customer_profile: dict
    evidence: Annotated[list[str], operator.add]
    agent_findings: Annotated[list[dict], operator.add]
    fraud_score: float
    decision: Literal["auto_credit", "deny", "human_review", "pending"]
    resolution_amount: float
    human_notes: str
    analyst_approved: bool | None
    notification_sent: bool
    error: str | None

    # ── Long-term memory (Phase 3.1 addition) ────────────────────────────────
    # Retrieved from ChromaDB before parallel checks run.
    # The aggregate node includes these in its LLM prompt — memory shapes scoring.
    memory_context: Annotated[list[str], operator.add]


# ═══════════════════════════════════════════════════════════════════════════════
# Intake node (unchanged from Phase 2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def intake(state: PersistenceState) -> dict:
    """Fetch transaction. Identical to Phase 2.2."""
    dispute = DISPUTES[state["dispute_id"]]
    txn = get_transaction.invoke({"transaction_id": dispute["transaction_id"]})
    return {
        "transaction": txn,
        "evidence": [
            f"Transaction {txn['id']}: ${txn['amount']:.2f} at {txn['merchant']} "
            f"({txn['location']}) on {txn['date']}"
        ],
        "decision": "pending",
        "fraud_score": 0.0,
        "resolution_amount": 0.0,
        "notification_sent": False,
        "error": None,
        "agent_findings": [],
        "memory_context": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Memory lookup node — Phase 3.1's first key addition
#
# Runs synchronously after intake, before parallel checks.
# Queries ChromaDB for any prior knowledge about this merchant and customer.
# Results are stored in memory_context and passed to the aggregate prompt.
# ═══════════════════════════════════════════════════════════════════════════════

def memory_lookup(state: PersistenceState) -> dict:
    """
    Query ChromaDB for prior knowledge relevant to this dispute.

    Two queries:
      1. Merchant query — any past outcomes for this merchant
      2. Customer query — any known behaviour patterns for this customer

    Results flow into memory_context, which the aggregate node uses to
    contextualise the parallel check signals before scoring.
    """
    txn = state["transaction"]
    merchant_id = txn.get("merchant_id", "")
    customer_id = state["customer_id"]
    merchant_name = txn.get("merchant", "")

    memories: list[str] = []

    try:
        collection = get_chroma_collection()
        total_docs = collection.count()

        if total_docs == 0:
            return {
                "memory_context": [],
                "evidence": ["Memory: collection is empty — cold run"],
            }

        # Cap n_results to avoid ChromaDB error when collection is smaller than requested
        n = max(1, min(3, total_docs))

        # Query 1: merchant history — semantic search, no where filter
        # (where filter on sparse metadata causes "Error finding id" in small collections)
        merchant_results = collection.query(
            query_texts=[f"merchant {merchant_name} {merchant_id} fraud dispute history"],
            n_results=n,
        )
        for doc in (merchant_results["documents"] or [[]])[0]:
            memories.append(f"[Merchant memory] {doc}")

        # Query 2: customer behaviour patterns — semantic search, no where filter
        customer_results = collection.query(
            query_texts=[f"customer {customer_id} travel spending behaviour pattern"],
            n_results=n,
        )
        for doc in (customer_results["documents"] or [[]])[0]:
            if doc not in memories:  # deduplicate if both queries return same doc
                memories.append(f"[Customer memory] {doc}")

    except Exception as e:
        memories.append(f"[Memory unavailable: {e}]")

    evidence = (
        [f"Memory: {len(memories)} prior knowledge item(s) retrieved from ChromaDB"]
        if memories
        else ["Memory: no prior knowledge found for this merchant/customer"]
    )

    return {
        "memory_context": memories,
        "evidence": evidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-out (identical to Phase 2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def fan_out(state: PersistenceState) -> list[Send]:
    """Dispatch 3 parallel checks via Send API."""
    return [
        Send("merchant_check", state),
        Send("customer_check", state),
        Send("velocity_check", state),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel check nodes (identical logic to Phase 2.2, no simulated latency)
# ═══════════════════════════════════════════════════════════════════════════════

def merchant_check(state: PersistenceState) -> dict:
    txn = state["transaction"]
    profile = check_merchant_fraud_history.invoke({"merchant_id": txn["merchant_id"]})

    risk_flags = []
    fraud_signal = 0.0

    if profile.get("known_skimmer"):
        risk_flags.append("KNOWN CARD SKIMMER")
        fraud_signal += 0.4
    if profile.get("prior_dispute_count", 0) > 20:
        risk_flags.append(f"{profile['prior_dispute_count']} prior disputes")
        fraud_signal += 0.3
    if profile.get("fraud_rate_pct", 0) > 15:
        risk_flags.append(f"{profile['fraud_rate_pct']}% fraud rate")
        fraud_signal += 0.2
    if not risk_flags:
        risk_flags.append(
            f"clean — {profile.get('prior_dispute_count', 0)} disputes, "
            f"{profile.get('fraud_rate_pct', 0)}% fraud rate"
        )
        fraud_signal -= 0.2

    summary = f"Merchant {profile.get('merchant_name')}: {'; '.join(risk_flags)}"
    return {
        "merchant_profile": profile,
        "agent_findings": [{"agent": "merchant_check", "summary": summary, "fraud_signal": fraud_signal, "flags": risk_flags}],
        "evidence": [f"[MerchantCheck] {summary}"],
    }


def customer_check(state: PersistenceState) -> dict:
    txn = state["transaction"]
    profile = get_customer_spend_history.invoke({"customer_id": state["customer_id"]})

    risk_flags = []
    fraud_signal = 0.0

    known_devices = profile.get("known_devices", [])
    if txn.get("device_id") and txn["device_id"] not in known_devices:
        risk_flags.append(f"unrecognised device {txn['device_id']}")
        fraud_signal += 0.2

    home_city = profile.get("home_city", "")
    recent_locs = {t.get("location", "") for t in profile.get("recent_transactions", [])}
    recent_locs.update([home_city, "Online"])
    if txn.get("location") and txn["location"] not in recent_locs:
        risk_flags.append(f"location anomaly: {txn['location']} vs home {home_city}")
        fraud_signal += 0.2

    avg = profile.get("avg_transaction_amount", 0)
    if avg > 0 and txn["amount"] > avg * 3:
        risk_flags.append(f"amount ${txn['amount']:.0f} is {txn['amount']/avg:.1f}× avg (${avg:.0f})")
        fraud_signal += 0.2

    merchant_name = txn.get("merchant", "").lower()
    recurring = [
        t for t in profile.get("recent_transactions", [])
        if merchant_name and merchant_name in t.get("merchant", "").lower()
    ]
    if recurring:
        risk_flags.append(f"recurring: {len(recurring)} prior charge(s) at same merchant")
        fraud_signal -= 0.4

    if not risk_flags:
        risk_flags.append("no anomalies — device, location, and amount are consistent")

    summary = f"Customer {state['customer_id']} ({home_city}): {'; '.join(risk_flags)}"
    return {
        "customer_profile": profile,
        "agent_findings": [{"agent": "customer_check", "summary": summary, "fraud_signal": fraud_signal, "flags": risk_flags}],
        "evidence": [f"[CustomerCheck] {summary}"],
    }


def velocity_check(state: PersistenceState) -> dict:
    result = check_velocity_pattern.invoke({"customer_id": state["customer_id"]})
    fraud_signal = 0.0
    flags = []

    if result.get("is_velocity_attack"):
        fraud_signal = 0.5
        flags.append(
            f"VELOCITY ATTACK: {result['charge_count']} charges of "
            f"${result['total_amount'] / result['charge_count']:.2f} "
            f"in {result['time_span_minutes']} min"
        )
    else:
        flags.append("no velocity attack detected")

    summary = "; ".join(flags)
    return {
        "agent_findings": [{"agent": "velocity_check", "summary": summary, "fraud_signal": fraud_signal, "flags": flags}],
        "evidence": [f"[VelocityCheck] {summary}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate node — memory-aware scoring
#
# The key difference from Phase 2.2: memory_context is injected into the
# LLM prompt. The model is explicitly told about prior knowledge (e.g. known
# travel patterns) and instructed to weigh it against the check signals.
# ═══════════════════════════════════════════════════════════════════════════════

class AggregateOutput(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_factors: list[str]


AGGREGATE_SYSTEM_PROMPT = """\
You are a senior fraud analyst. You receive:
  1. Structured findings from 3 parallel specialist checks
  2. Prior knowledge retrieved from long-term memory (if any)

Each check finding includes a fraud_signal (positive = fraud evidence, \
negative = legitimacy evidence).

Important: prior knowledge from memory takes precedence over raw signal values. \
If memory confirms a behaviour pattern is legitimate (e.g. a customer's known \
travel route), reduce the weight of the corresponding anomaly signal accordingly. \
If memory confirms a merchant is a confirmed fraud hotspot, amplify that signal.

Produce a final fraud_score (0.0 = certainly legitimate, 1.0 = certainly fraud).
"""


async def aggregate(state: PersistenceState) -> dict:
    """
    Convergence node. Runs after all parallel checks complete.
    Incorporates memory_context into the LLM prompt — memory shapes the score.
    """
    findings = state.get("agent_findings", [])
    memory = state.get("memory_context", [])
    txn = state["transaction"]

    findings_text = "\n".join(
        f"  {f['agent']} (signal: {f['fraud_signal']:+.1f}): {f['summary']}"
        for f in findings
    )

    memory_text = (
        "\n".join(f"  - {m}" for m in memory)
        if memory
        else "  (none — running cold, no prior knowledge available)"
    )

    prompt = f"""
Dispute: {state['dispute_id']}
Transaction: ${txn['amount']:.2f} at {txn['merchant']} ({txn['location']})

Parallel check results:
{findings_text}

Prior knowledge from long-term memory:
{memory_text}

Compute the composite fraud_score, accounting for memory context. \
List key_factors and provide reasoning.
""".strip()

    llm = get_investigator(max_tokens=512)
    result: AggregateOutput = await llm.with_structured_output(AggregateOutput).ainvoke([
        SystemMessage(content=AGGREGATE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "fraud_score": result.fraud_score,
        "human_notes": result.reasoning,
        "evidence": result.key_factors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome nodes (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def auto_credit(state: PersistenceState) -> dict:
    txn = state["transaction"]
    return {
        "decision": "auto_credit",
        "resolution_amount": txn["amount"],
        "evidence": [f"AUTO-CREDIT ${txn['amount']:.2f} — score {state['fraud_score']:.2f}"],
    }

def deny(state: PersistenceState) -> dict:
    return {
        "decision": "deny",
        "resolution_amount": 0.0,
        "evidence": [f"DENY — score {state['fraud_score']:.2f}, transaction appears legitimate"],
    }

def human_review(state: PersistenceState) -> dict:
    txn = state["transaction"]
    reason = (
        f"amount ${txn['amount']:.2f} ≥ $500"
        if txn["amount"] >= 500
        else f"ambiguous score {state['fraud_score']:.2f}"
    )
    return {
        "decision": "human_review",
        "resolution_amount": txn["amount"],
        "evidence": [f"HUMAN REVIEW — {reason}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Store outcome node — Phase 3.1's second key addition
#
# After each dispute resolves, embed the outcome into ChromaDB.
# Future disputes on the same merchant or customer will retrieve this
# as memory context, progressively improving accuracy over time.
# ═══════════════════════════════════════════════════════════════════════════════

def store_outcome(state: PersistenceState) -> dict:
    """
    Persist the dispute outcome to ChromaDB long-term memory.

    Stores two documents per dispute:
      1. Merchant outcome — what happened at this merchant
      2. Customer outcome — what we learned about this customer's behaviour

    These become the memory_context for future disputes.
    """
    txn = state["transaction"]
    dispute_id = state["dispute_id"]
    customer_id = state["customer_id"]
    merchant_id = txn.get("merchant_id", "")
    decision = state["decision"]
    score = state["fraud_score"]

    try:
        collection = get_chroma_collection()

        # 1. Merchant outcome document
        merchant_doc = (
            f"Dispute {dispute_id}: ${txn['amount']:.2f} at {txn['merchant']} "
            f"({txn['location']}) — resolved as {decision.upper()} "
            f"(fraud_score: {score:.2f}). "
            f"Evidence: {'; '.join(state.get('evidence', [])[:3])}"
        )
        collection.upsert(
            ids=[f"{dispute_id}_merchant"],
            documents=[merchant_doc],
            metadatas=[{
                "dispute_id": dispute_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "decision": decision,
                "fraud_score": score,
                "type": "merchant_outcome",
            }],
        )

        # 2. Customer behaviour document
        customer_doc = (
            f"Customer {customer_id} dispute {dispute_id}: "
            f"transaction at {txn['merchant']} in {txn['location']} "
            f"on {txn.get('date', 'unknown')} — resolved as {decision.upper()} "
            f"(fraud_score: {score:.2f})"
        )
        collection.upsert(
            ids=[f"{dispute_id}_customer"],
            documents=[customer_doc],
            metadatas=[{
                "dispute_id": dispute_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "decision": decision,
                "fraud_score": score,
                "type": "customer_outcome",
            }],
        )

        stored_count = 2
    except Exception as e:
        stored_count = 0
        console.print(f"[dim red]Warning: could not store outcome in ChromaDB: {e}[/dim red]")

    return {
        "evidence": [
            f"Memory: {stored_count} outcome document(s) stored in ChromaDB for future disputes"
        ],
    }


def notify(state: PersistenceState) -> dict:
    decision = state["decision"]
    txn = state["transaction"]
    messages = {
        "auto_credit":  f"Dispute approved — ${state['resolution_amount']:.2f} credit issued for {txn['merchant']}.",
        "deny":         f"Dispute denied — ${txn['amount']:.2f} at {txn['merchant']} appears legitimate.",
        "human_review": f"Dispute under review — a specialist will contact you within 2 business days.",
    }
    return {
        "notification_sent": True,
        "evidence": [f"Notification: {messages.get(decision, '')}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════════

def decide_route(state: PersistenceState) -> Literal["auto_credit", "deny", "human_review"]:
    score = state["fraud_score"]
    amount = state["transaction"]["amount"]
    if score > 0.8 and amount < 500:
        return "auto_credit"
    elif score < 0.4:
        return "deny"
    else:
        return "human_review"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(checkpointer=None):
    g = StateGraph(PersistenceState)

    g.add_node("intake",          intake)
    g.add_node("memory_lookup",   memory_lookup)
    g.add_node("merchant_check",  merchant_check)
    g.add_node("customer_check",  customer_check)
    g.add_node("velocity_check",  velocity_check)
    g.add_node("aggregate",       aggregate)
    g.add_node("auto_credit",     auto_credit)
    g.add_node("deny",            deny)
    g.add_node("human_review",    human_review)
    g.add_node("store_outcome",   store_outcome)
    g.add_node("notify",          notify)

    # intake → memory_lookup → fan_out → [parallel checks] → aggregate
    g.set_entry_point("intake")
    g.add_edge("intake", "memory_lookup")
    g.add_conditional_edges("memory_lookup", fan_out)

    # All parallel branches converge at aggregate
    g.add_edge("merchant_check", "aggregate")
    g.add_edge("customer_check", "aggregate")
    g.add_edge("velocity_check", "aggregate")

    # aggregate → decide → outcome → store_outcome → notify → END
    g.add_conditional_edges(
        "aggregate",
        decide_route,
        {"auto_credit": "auto_credit", "deny": "deny", "human_review": "human_review"},
    )
    g.add_edge("auto_credit",  "store_outcome")
    g.add_edge("deny",         "store_outcome")
    g.add_edge("human_review", "store_outcome")
    g.add_edge("store_outcome","notify")
    g.add_edge("notify",       END)

    if not isinstance(checkpointer, BaseCheckpointSaver):
        checkpointer = None
    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# Memory seeding utilities
# ═══════════════════════════════════════════════════════════════════════════════

def seed_memory():
    """
    Seed ChromaDB with known customer travel patterns and merchant context.
    This simulates the long-term memory that would accumulate after weeks of
    real dispute processing.

    Run this between the cold and warm runs to see memory change outcomes.
    """
    collection = get_chroma_collection()

    seeds = [
        # Customer travel patterns
        {
            "id": "seed_cust4821_miami_travel",
            "document": (
                "Customer CUST-4821 (Sarah Chen, Seattle WA) travels to Miami, FL "
                "every April for the annual FinTech conference. This is a verified "
                "recurring travel pattern — Miami transactions in April should NOT "
                "be treated as location anomalies for this customer."
            ),
            "metadata": {
                "customer_id": "CUST-4821",
                "merchant_id": "",
                "type": "customer_pattern",
                "decision": "legitimate",
                "fraud_score": 0.0,
                "dispute_id": "seed",
            },
        },
        # Merchant context (separately seeded — would come from analyst feedback)
        {
            "id": "seed_mch0042_history",
            "document": (
                "Merchant MCH-0042 (Biscayne Grill, Miami FL) has 47 confirmed fraud "
                "disputes on record. High-risk merchant in a tourist area. However, "
                "legitimate customers do dine here — cross-reference customer travel "
                "patterns before auto-crediting."
            ),
            "metadata": {
                "customer_id": "",
                "merchant_id": "MCH-0042",
                "type": "merchant_pattern",
                "decision": "high_risk",
                "fraud_score": 0.7,
                "dispute_id": "seed",
            },
        },
    ]

    for seed in seeds:
        collection.upsert(
            ids=[seed["id"]],
            documents=[seed["document"]],
            metadatas=[seed["metadata"]],
        )

    console.print(Panel.fit(
        f"[bold green]Memory seeded[/bold green] — {len(seeds)} documents stored in ChromaDB\n\n"
        + "\n".join(f"  [dim]• {s['document'][:90]}...[/dim]" for s in seeds),
        title="ChromaDB Seed",
        border_style="green",
    ))


def clear_memory():
    """Delete and recreate the collection — fresh start for cold run demo."""
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    console.print("[dim]ChromaDB memory cleared — cold run[/dim]\n")


def show_memory():
    """Display all documents currently stored in ChromaDB."""
    collection = get_chroma_collection()
    result = collection.get()
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    ids = result.get("ids", [])

    if not docs:
        console.print("[dim]No memories stored yet.[/dim]")
        return

    table = Table(title=f"ChromaDB Memory — {len(docs)} document(s)", box=box.SIMPLE)
    table.add_column("ID", style="dim", width=30)
    table.add_column("Type", width=18)
    table.add_column("Document (truncated)")

    for doc_id, doc, meta in zip(ids, docs, metas):
        table.add_row(
            doc_id,
            meta.get("type", "—"),
            doc[:80] + ("..." if len(doc) > 80 else ""),
        )
    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "intake":         "blue",
    "memory_lookup":  "bright_magenta",
    "merchant_check": "cyan",
    "customer_check": "cyan",
    "velocity_check": "magenta",
    "aggregate":      "yellow",
    "auto_credit":    "green",
    "deny":           "red",
    "human_review":   "yellow",
    "store_outcome":  "bright_magenta",
    "notify":         "white",
}
DECISION_COLORS = {
    "auto_credit": "green", "deny": "red",
    "human_review": "yellow", "pending": "dim",
}
PARALLEL_NODES = {"merchant_check", "customer_check", "velocity_check"}


async def run_dispute(app, dispute_id: str, thread_suffix: str = ""):
    dispute = DISPUTES[dispute_id]
    initial: PersistenceState = {
        "dispute_id":       dispute_id,
        "customer_id":      dispute["customer_id"],
        "transaction":      {},
        "merchant_profile": {},
        "customer_profile": {},
        "evidence":         [],
        "agent_findings":   [],
        "memory_context":   [],
        "fraud_score":      0.0,
        "decision":         "pending",
        "resolution_amount":0.0,
        "human_notes":      "",
        "analyst_approved": None,
        "notification_sent":False,
        "error":            None,
    }
    # thread_id: same ID = resume from checkpoint; different = fresh run
    thread_id = f"3_1_{dispute_id}{thread_suffix}"
    config = {"configurable": {"thread_id": thread_id}}

    console.print(f"\n[bold]Dispute {dispute_id}[/bold]  customer: {dispute['customer_id']}  thread: {thread_id}")
    console.print("─" * 70)

    parallel_completed: list[str] = []
    t0 = time.perf_counter()

    async for update in app.astream(initial, config=config, stream_mode="updates"):
        for node_name, changes in update.items():
            color = NODE_COLORS.get(node_name, "white")
            extras = []

            if node_name == "intake":
                txn = changes.get("transaction", {})
                extras.append(f"${txn['amount']:.2f} at {txn['merchant']}")

            elif node_name == "memory_lookup":
                memories = changes.get("memory_context", [])
                if memories:
                    extras.append(f"[bold bright_magenta]{len(memories)} memory item(s) retrieved[/bold bright_magenta]")
                else:
                    extras.append("[dim]no prior knowledge[/dim]")

            elif node_name in PARALLEL_NODES:
                parallel_completed.append(node_name)
                finding = next(
                    (f for f in changes.get("agent_findings", []) if f["agent"] == node_name), None
                )
                if finding:
                    signal = finding["fraud_signal"]
                    sc = "red" if signal > 0 else ("green" if signal < 0 else "dim")
                    extras.append(f"signal [{sc}]{signal:+.1f}[/{sc}]")

            elif node_name == "aggregate" and "fraud_score" in changes:
                score = changes["fraud_score"]
                sc = "green" if score < 0.4 else ("red" if score > 0.8 else "yellow")
                extras.append(f"score → [bold {sc}]{score:.2f}[/bold {sc}]")

            elif node_name in ("auto_credit", "deny", "human_review"):
                d = changes.get("decision", node_name)
                dc = DECISION_COLORS.get(d, "white")
                extras.append(f"[bold {dc}]{d.upper().replace('_', ' ')}[/bold {dc}]")

            elif node_name == "store_outcome":
                extras.append("outcome stored to ChromaDB ✓")

            elif node_name == "notify":
                extras.append("notification sent ✓")

            console.print(
                f"  [{color}]■[/{color}] [bold {color}]{node_name:<18}[/bold {color}]"
                + (f"  {'  '.join(extras)}" if extras else "")
            )

            # Print memory context items inline
            if node_name == "memory_lookup":
                for mem in changes.get("memory_context", []):
                    console.print(f"               [dim bright_magenta]↳ {mem[:100]}[/dim bright_magenta]")
            else:
                for item in changes.get("evidence", []):
                    console.print(f"               [dim]↳ {item}[/dim]")

    total_elapsed = time.perf_counter() - t0
    final = (await app.aget_state(config)).values
    _print_final(dispute_id, final, total_elapsed)


def _print_final(dispute_id: str, state: dict, elapsed: float):
    decision = state.get("decision", "pending")
    color = DECISION_COLORS.get(decision, "white")

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("k", style="dim", width=22)
    table.add_column("v")
    table.add_row("Fraud score",    f"[bold]{state.get('fraud_score', 0):.2f}[/bold]")
    table.add_row("Decision",       f"[bold {color}]{decision.upper().replace('_', ' ')}[/bold {color}]")
    amt = state.get("resolution_amount", 0)
    table.add_row("Resolution",     f"${amt:.2f}" if amt > 0 else "—")
    table.add_row("Memory items",   str(len(state.get("memory_context", []))))
    table.add_row("Evidence items", str(len(state.get("evidence", []))))
    table.add_row("Total elapsed",  f"{elapsed:.2f}s")

    console.print(Panel(table, title=f"[bold]{dispute_id} Final State[/bold]", border_style=color))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispute", choices=list(DISPUTES.keys()))
    parser.add_argument("--warm",        action="store_true", help="Run with existing ChromaDB memory (don't clear)")
    parser.add_argument("--seed-memory", action="store_true", help="Seed ChromaDB with customer travel patterns, then exit")
    parser.add_argument("--show-memory", action="store_true", help="Display all ChromaDB documents, then exit")
    parser.add_argument("--resume",      action="store_true", help="Re-use same thread_id (demonstrates resumability from SQLite checkpoint)")
    args = parser.parse_args()

    cfg = current_config()
    console.print(Panel.fit(
        f"[bold cyan]Phase 3.1 — Persistence & Long-term Memory[/bold cyan]\n"
        f"Short-term: SqliteSaver (resumable threads) · "
        f"Long-term: ChromaDB (cross-dispute learning)\n"
        f"[dim]provider: {cfg['provider']}  investigator: {cfg['investigator']}[/dim]",
        border_style="cyan",
    ))

    if args.show_memory:
        show_memory()
        return

    if args.seed_memory:
        seed_memory()
        return

    if not args.warm:
        clear_memory()

    # thread_suffix: empty = fresh thread each run; "_resume" = same thread (resume demo)
    thread_suffix = "" if not args.resume else "_resume"

    async with AsyncSqliteSaver.from_conn_string("disputes.db") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        try:
            png_path = "phase3_1_persistence/graph.png"
            app.get_graph().draw_mermaid_png(output_file_path=png_path)
            console.print(f"[dim]Graph saved → {png_path}[/dim]\n")
        except Exception:
            console.print("[dim]Graph PNG skipped[/dim]\n")

        dispute_ids = [args.dispute] if args.dispute else list(DISPUTES.keys())
        for dispute_id in dispute_ids:
            await run_dispute(app, dispute_id, thread_suffix=thread_suffix)

        console.print(Rule("[dim]All disputes complete[/dim]"))


if __name__ == "__main__":
    asyncio.run(main())
