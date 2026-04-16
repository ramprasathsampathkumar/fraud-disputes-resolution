"""
Phase 1.2 — LangGraph State Machine

Concepts demonstrated:
  - StateGraph with typed DisputeState
  - Nodes as pure functions that return partial state updates
  - Annotated[list, operator.add] — safe evidence accumulation across nodes
  - Conditional edges — deterministic routing based on fraud_score + amount
  - SqliteSaver checkpointer — every node transition persisted to disk
  - stream_mode="updates" — watch state evolve node-by-node in real time
  - Graph visualisation via Mermaid

Graph structure:
    intake → enrich → score → [decide_route] ─┬→ auto_credit ─┐
                                               ├→ human_review ─┤→ notify
                                               └→ deny ─────────┘

Key design principles applied from Phase 1.1 findings:
  - Score node uses Gather-then-Reason (1 LLM call, all data already in state)
  - All other nodes are pure deterministic functions — zero LLM tokens
  - LLM is called exactly once per dispute, only for fraud reasoning

Run:
    uv run python -m phase1_2_state_machine.graph
    uv run python -m phase1_2_state_machine.graph --dispute DISP-004
"""

import argparse
import asyncio
import json
import logging
import os
import warnings
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

load_dotenv()

warnings.filterwarnings("ignore", message="create_react_agent has been moved", category=DeprecationWarning)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from shared.models import DisputeState
from shared.mock_data import DISPUTES
from shared.model_factory import get_investigator, current_config
from shared.tools import (
    get_transaction,
    check_merchant_fraud_history,
    get_customer_spend_history,
    check_velocity_pattern,
)

console = Console()

# ── LangSmith guard ──────────────────────────────────────────────────────────
_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if _ls_key and not _ls_key.startswith(("ls__", "lsv2_")):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ═══════════════════════════════════════════════════════════════════════════════
# Structured output for the score node — lighter than full FraudDecision
# ═══════════════════════════════════════════════════════════════════════════════

class ScoreOutput(BaseModel):
    """
    What the LLM produces in the score node.
    Deliberately narrower than FraudDecision — the LLM's job here is only to
    assess the evidence and assign a score. Routing and resolution are handled
    deterministically by subsequent nodes.
    """
    fraud_score: float = Field(
        ge=0.0, le=1.0,
        description="Fraud probability: 0.0 = certainly legitimate, 1.0 = certainly fraud"
    )
    evidence_items: list[str] = Field(
        description="Key findings, one per item. Each should cite a specific data point."
    )
    reasoning: str = Field(
        description="One-paragraph explanation of how the signals led to this score."
    )


SCORE_SYSTEM_PROMPT = """\
You are a fraud analyst. All transaction evidence is provided below.
Compute a fraud_score between 0.0 and 1.0 using these additive signals:

  +0.3  Merchant > 20 prior disputes
  +0.2  Merchant fraud rate > 15%
  +0.2  Transaction location not in customer's known locations
  +0.2  Unrecognised device (not in customer's known_devices)
  +0.2  Amount > 3× customer's average transaction
  +0.4  Known card skimmer at merchant
  +0.5  Velocity attack (3+ identical charges within 2 hours)
  -0.4  Customer has prior recurring/subscription charge at same merchant
  -0.2  Merchant has 0 disputes and < 1% fraud rate

Return ONLY the fraud_score, evidence_items, and reasoning. Do not decide on credit/deny — that is handled downstream.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════════

def intake(state: DisputeState) -> dict:
    """
    Node 1 — Intake
    Fetches the raw transaction record and seeds the evidence list.
    Pure data fetch — zero LLM tokens.
    """
    dispute = DISPUTES[state["dispute_id"]]
    txn = get_transaction.invoke({"transaction_id": dispute["transaction_id"]})

    return {
        "transaction": txn,
        # Annotated[list, operator.add] means this list is APPENDED to state,
        # not replaced. Every node safely adds its own evidence items.
        "evidence": [
            f"Transaction {txn['id']}: ${txn['amount']:.2f} at {txn['merchant']} "
            f"({txn['location']}) on {txn['date']}"
        ],
        "decision": "pending",
        "fraud_score": 0.0,
        "resolution_amount": 0.0,
        "notification_sent": False,
        "error": None,
    }


def enrich(state: DisputeState) -> dict:
    """
    Node 2 — Enrich
    Loads merchant risk profile, customer spend history, and velocity data.
    Appends preliminary evidence flags. Pure data fetch — zero LLM tokens.
    """
    txn = state["transaction"]
    merchant = check_merchant_fraud_history.invoke({"merchant_id": txn["merchant_id"]})
    customer = get_customer_spend_history.invoke({"customer_id": state["customer_id"]})
    velocity = check_velocity_pattern.invoke({"customer_id": state["customer_id"]})

    evidence = []

    # Merchant signals
    if merchant.get("known_skimmer"):
        evidence.append(f"ALERT: Known card skimmer at {txn['merchant']} (merchant {txn['merchant_id']})")
    if merchant.get("prior_dispute_count", 0) > 20:
        evidence.append(
            f"High-risk merchant: {merchant['prior_dispute_count']} prior disputes, "
            f"{merchant['fraud_rate_pct']}% fraud rate"
        )
    elif merchant.get("prior_dispute_count", 0) == 0:
        evidence.append(f"Clean merchant: 0 prior disputes, {merchant['fraud_rate_pct']}% fraud rate")

    # Customer / device signals
    known_devices = customer.get("known_devices", [])
    if txn.get("device_id") and txn["device_id"] not in known_devices:
        evidence.append(f"Unrecognised device: {txn['device_id']} (known: {known_devices})")

    # Location signal
    home_city = customer.get("home_city", "")
    recent_locations = {t.get("location", "") for t in customer.get("recent_transactions", [])}
    recent_locations.add(home_city)
    recent_locations.add("Online")
    if txn.get("location") and txn["location"] not in recent_locations:
        evidence.append(
            f"Location anomaly: {txn['location']} — customer home is {home_city}, "
            f"recent activity in {sorted(recent_locations - {'Online'})}"
        )

    # Amount signal
    avg = customer.get("avg_transaction_amount", 0)
    if avg > 0 and txn["amount"] > avg * 3:
        evidence.append(
            f"Amount {txn['amount']:.2f} is {txn['amount']/avg:.1f}× customer average ({avg:.2f})"
        )

    # Velocity signal
    if velocity.get("is_velocity_attack"):
        evidence.append(
            f"VELOCITY ATTACK: {velocity['charge_count']} charges of "
            f"${velocity['total_amount']/velocity['charge_count']:.2f} "
            f"in {velocity['time_span_minutes']} minutes (total ${velocity['total_amount']:.2f})"
        )

    # Subscription/recurring signal
    merchant_name = merchant.get("merchant_name", "").lower()
    prior_txns = customer.get("recent_transactions", [])
    recurring = [t for t in prior_txns if merchant_name in t.get("merchant", "").lower()]
    if recurring:
        evidence.append(
            f"Recurring pattern: customer has {len(recurring)} prior charge(s) at {txn['merchant']}"
        )

    return {
        "merchant_profile": merchant,
        "customer_profile": {**customer, "velocity": velocity},
        "evidence": evidence,
    }


async def score(state: DisputeState) -> dict:
    """
    Node 3 — Score
    The only node that calls the LLM. Uses Gather-then-Reason:
    all data is already in state, so this is exactly ONE LLM call.
    """
    txn      = state["transaction"]
    merchant = state["merchant_profile"]
    customer = state["customer_profile"]

    evidence_block = "\n".join(f"  - {e}" for e in state["evidence"])

    prompt = f"""
DISPUTE: {state['dispute_id']}

TRANSACTION:
  Amount: ${txn['amount']:.2f}
  Merchant: {txn['merchant']} (ID: {txn['merchant_id']})
  Location: {txn['location']}
  Date: {txn['date']}
  Device: {txn.get('device_id', 'unknown')}

MERCHANT RISK:
  Prior disputes: {merchant.get('prior_dispute_count', 0)}
  Fraud rate: {merchant.get('fraud_rate_pct', 0)}%
  Known skimmer: {merchant.get('known_skimmer', False)}
  Flags: {merchant.get('flagged_categories', [])}

CUSTOMER PROFILE:
  Home city: {customer.get('home_city')}
  Known devices: {customer.get('known_devices', [])}
  Avg transaction: ${customer.get('avg_transaction_amount', 0):.2f}
  Velocity attack: {customer.get('velocity', {}).get('is_velocity_attack', False)}

PRE-COMPUTED EVIDENCE FLAGS:
{evidence_block}

Compute the fraud_score and list your evidence_items and reasoning.
""".strip()

    llm = get_investigator(max_tokens=512)
    llm_with_output = llm.with_structured_output(ScoreOutput)

    result: ScoreOutput = await llm_with_output.ainvoke([
        SystemMessage(content=SCORE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "fraud_score": result.fraud_score,
        # These append to existing evidence via the operator.add reducer
        "evidence": result.evidence_items,
        "human_notes": result.reasoning,
    }


def auto_credit(state: DisputeState) -> dict:
    """Node 4a — high confidence fraud, amount under threshold → immediate credit."""
    txn = state["transaction"]
    return {
        "decision": "auto_credit",
        "resolution_amount": txn["amount"],
        "evidence": [f"Decision: AUTO-CREDIT ${txn['amount']:.2f} — fraud score {state['fraud_score']:.2f} exceeds threshold"],
    }


def deny(state: DisputeState) -> dict:
    """Node 4b — low fraud score → dispute denied, transaction stands."""
    return {
        "decision": "deny",
        "resolution_amount": 0.0,
        "evidence": [f"Decision: DENY — fraud score {state['fraud_score']:.2f} below threshold, transaction appears legitimate"],
    }


def human_review(state: DisputeState) -> dict:
    """Node 4c — ambiguous or high-value → route to analyst with evidence brief."""
    txn = state["transaction"]
    score = state["fraud_score"]
    amount = txn["amount"]

    reason = (
        f"amount ${amount:.2f} ≥ $500 threshold"
        if amount >= 500
        else f"ambiguous fraud score {score:.2f} (0.4–0.8 range)"
    )

    return {
        "decision": "human_review",
        "resolution_amount": txn["amount"],  # pending analyst approval
        "evidence": [f"Routed to human review: {reason}"],
    }


def notify(state: DisputeState) -> dict:
    """
    Node 5 — Notify
    Final node. In production this would send an email/push notification.
    Here we log the outcome and mark the dispute complete.
    """
    decision = state["decision"]
    txn = state["transaction"]

    messages = {
        "auto_credit": (
            f"Your dispute for ${txn['amount']:.2f} at {txn['merchant']} has been approved. "
            f"A credit of ${state['resolution_amount']:.2f} will appear within 1–2 business days."
        ),
        "deny": (
            f"After investigation, your dispute for ${txn['amount']:.2f} at {txn['merchant']} "
            f"was not approved. The transaction appears to be legitimate."
        ),
        "human_review": (
            f"Your dispute for ${txn['amount']:.2f} at {txn['merchant']} requires additional review. "
            f"A specialist will contact you within 2 business days."
        ),
    }

    return {
        "notification_sent": True,
        "evidence": [f"Notification sent: {messages.get(decision, 'Unknown decision')}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routing function (conditional edge — not a node)
# ═══════════════════════════════════════════════════════════════════════════════

def decide_route(state: DisputeState) -> Literal["auto_credit", "deny", "human_review"]:
    """
    Pure deterministic routing — no LLM involved.
    This is a conditional edge function, not a node. It reads state and returns
    the name of the next node to execute.

    Senior note: keeping routing deterministic (not LLM-driven) is intentional.
    Routing logic belongs in code, not in prompts — it's auditable, testable,
    and never hallucinates.
    """
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
    """
    Assemble and compile the StateGraph.

    Passing a checkpointer enables persistence — every node transition is
    written to SQLite. The dispute can be resumed from any checkpoint using
    the same thread_id (dispute_id).
    """
    g = StateGraph(DisputeState)

    # Register nodes
    g.add_node("intake",       intake)
    g.add_node("enrich",       enrich)
    g.add_node("score",        score)        # async node — LLM call
    g.add_node("auto_credit",  auto_credit)
    g.add_node("deny",         deny)
    g.add_node("human_review", human_review)
    g.add_node("notify",       notify)

    # Linear edges
    g.set_entry_point("intake")
    g.add_edge("intake", "enrich")
    g.add_edge("enrich", "score")

    # Conditional edge: score → decide_route → one of three outcome nodes
    g.add_conditional_edges(
        "score",
        decide_route,
        {
            "auto_credit":  "auto_credit",
            "deny":         "deny",
            "human_review": "human_review",
        },
    )

    # All outcome nodes converge on notify
    g.add_edge("auto_credit",  "notify")
    g.add_edge("deny",         "notify")
    g.add_edge("human_review", "notify")
    g.add_edge("notify",       END)

    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI runner with streaming state display
# ═══════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "intake":       "blue",
    "enrich":       "cyan",
    "score":        "magenta",
    "auto_credit":  "green",
    "deny":         "red",
    "human_review": "yellow",
    "notify":       "white",
}

DECISION_COLORS = {
    "auto_credit":  "green",
    "deny":         "red",
    "human_review": "yellow",
    "pending":      "dim",
}


async def run_dispute(app, dispute_id: str):
    """
    Stream a single dispute through the graph, printing state after each node.
    stream_mode='updates' yields {node_name: state_changes} after each node completes.
    """
    dispute = DISPUTES[dispute_id]

    initial_state: DisputeState = {
        "dispute_id":       dispute_id,
        "customer_id":      dispute["customer_id"],
        "transaction":      {},
        "merchant_profile": {},
        "customer_profile": {},
        "evidence":         [],
        "fraud_score":      0.0,
        "decision":         "pending",
        "resolution_amount": 0.0,
        "human_notes":      "",
        "analyst_approved": None,
        "notification_sent": False,
        "error":            None,
    }

    # thread_id scopes the checkpoint — same dispute_id always resumes the same run
    config = {"configurable": {"thread_id": dispute_id}}

    console.print(f"\n[bold]Dispute {dispute_id}[/bold]  customer: {dispute['customer_id']}")
    console.print("─" * 70)

    accumulated_evidence: list[str] = []

    async for update in app.astream(initial_state, config=config, stream_mode="updates"):
        for node_name, changes in update.items():
            color = NODE_COLORS.get(node_name, "white")
            new_evidence = changes.get("evidence", [])
            accumulated_evidence.extend(new_evidence)

            # Build a compact status line for each node
            extras = []
            if "transaction" in changes and changes["transaction"]:
                txn = changes["transaction"]
                extras.append(f"${txn['amount']:.2f} at {txn['merchant']}")
            if "fraud_score" in changes and changes["fraud_score"] > 0:
                extras.append(f"score → [bold]{changes['fraud_score']:.2f}[/bold]")
            if "decision" in changes and changes["decision"] != "pending":
                d = changes["decision"]
                dc = DECISION_COLORS.get(d, "white")
                extras.append(f"[bold {dc}]{d.upper().replace('_',' ')}[/bold {dc}]")
            if "notification_sent" in changes and changes["notification_sent"]:
                extras.append("notification sent")

            extra_str = "  " + " | ".join(extras) if extras else ""
            console.print(
                f"  [{color}]■[/{color}] [bold {color}]{node_name:<14}[/bold {color}]"
                f"{extra_str}"
            )

            # Print new evidence items indented under the node
            for item in new_evidence:
                console.print(f"             [dim]↳ {item}[/dim]")

    # Final state summary
    final = (await app.aget_state(config)).values
    _print_final(dispute_id, final)


def _print_final(dispute_id: str, state: dict):
    decision = state.get("decision", "pending")
    color = DECISION_COLORS.get(decision, "white")

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("k", style="dim", width=18)
    table.add_column("v")

    table.add_row("Fraud score", f"[bold]{state.get('fraud_score', 0):.2f}[/bold]")
    table.add_row(
        "Decision",
        f"[bold {color}]{decision.upper().replace('_', ' ')}[/bold {color}]",
    )
    amt = state.get("resolution_amount", 0)
    table.add_row("Resolution", f"${amt:.2f}" if amt > 0 else "—")
    table.add_row("Evidence items", str(len(state.get("evidence", []))))
    table.add_row("Checkpointed", "✓ disputes.db")

    console.print(Panel(table, title=f"[bold]{dispute_id} Final State[/bold]", border_style=color))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispute", choices=list(DISPUTES.keys()),
        help="Run a single dispute (default: all 4)",
    )
    args = parser.parse_args()

    cfg = current_config()
    console.print(Panel.fit(
        f"[bold cyan]Phase 1.2 — LangGraph State Machine[/bold cyan]\n"
        f"Graph: intake → enrich → score → [decide] → outcome → notify\n"
        f"[dim]provider: {cfg['provider']}  model: {cfg['investigator']}  "
        f"checkpointer: AsyncSqliteSaver (disputes.db)[/dim]",
        border_style="cyan",
    ))

    # Print the graph structure as Mermaid — useful during development
    async with AsyncSqliteSaver.from_conn_string("disputes.db") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        console.print("\n[dim]Graph structure (Mermaid):[/dim]")
        console.print(f"[dim]{app.get_graph().draw_mermaid()}[/dim]")

        try:
            png_path = "phase1_2_state_machine/graph.png"
            app.get_graph().draw_mermaid_png(output_file_path=png_path)
            console.print(f"[dim]Graph saved → [bold]{png_path}[/bold] (open to view)[/dim]")
        except Exception:
            console.print("[dim]Graph PNG skipped (mermaid.ink unreachable — paste Mermaid output into mermaid.live)[/dim]")

        dispute_ids = [args.dispute] if args.dispute else list(DISPUTES.keys())

        for dispute_id in dispute_ids:
            await run_dispute(app, dispute_id)

        console.print(Rule("[dim]All disputes complete[/dim]"))


if __name__ == "__main__":
    asyncio.run(main())
