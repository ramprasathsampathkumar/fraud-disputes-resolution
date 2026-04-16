"""
Phase 2.1 — Supervisor Pattern

Concepts demonstrated:
  - Supervisor node that orchestrates specialist agents in a loop
  - Each specialist agent has a single focused responsibility
  - Supervisor uses LLM (cheap classifier model) to decide next agent
  - Conditional routing: DISP-004 gets VelocityAgent; others skip it
  - Specialist agents are deterministic (no LLM) — LLM reserved for routing + scoring
  - State accumulation: each agent appends its findings to shared state

Graph structure:
                     ┌─────────────────────────┐
                     ↓                         │ (loop back)
    intake → supervisor ──→ merchant_intel ────┘
                     │
                     ├──→ customer_profile ────┐
                     │                         │ (loop back)
                     │                         ↓
                     ├──→ velocity_agent  ────→ supervisor
                     │
                     └──→ aggregate → [decide] → outcome → notify

Key design decisions vs Phase 1.2:
  - Specialist agents have FOCUSED responsibilities — merchant_intel knows nothing
    about customer profiles, velocity_agent knows nothing about merchant history
  - Supervisor uses the cheap classifier model for routing decisions
  - Aggregate node makes the single LLM reasoning call (same Gather-then-Reason pattern)
  - VelocityAgent is only invoked when warranted — supervisor decides dynamically

Run:
    uv run python -m phase2_1_supervisor.graph
    uv run python -m phase2_1_supervisor.graph --dispute DISP-004
"""

import argparse
import asyncio
import json
import logging
import os
import warnings
from typing import Annotated, Literal
import operator

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box
from typing_extensions import TypedDict

load_dotenv()

warnings.filterwarnings("ignore", message="create_react_agent has been moved", category=DeprecationWarning)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from shared.mock_data import DISPUTES
from shared.model_factory import get_investigator, get_classifier, current_config
from shared.tools import (
    get_transaction,
    check_merchant_fraud_history,
    get_customer_spend_history,
    check_velocity_pattern,
)

console = Console()

_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if _ls_key and not _ls_key.startswith(("ls__", "lsv2_")):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


# ═══════════════════════════════════════════════════════════════════════════════
# State schema
# Extends Phase 1.2's DisputeState with supervisor-specific fields.
# ═══════════════════════════════════════════════════════════════════════════════

class SupervisorState(TypedDict):
    # ── Core dispute fields (same as DisputeState) ──────────────────────────
    dispute_id: str
    customer_id: str
    transaction: dict
    merchant_profile: dict
    customer_profile: dict
    evidence: Annotated[list[str], operator.add]
    fraud_score: float
    decision: Literal["auto_credit", "deny", "human_review", "pending"]
    resolution_amount: float
    human_notes: str
    analyst_approved: bool | None
    notification_sent: bool
    error: str | None

    # ── Supervisor-specific fields ───────────────────────────────────────────
    # Which agent the supervisor has chosen to run next
    next_agent: str

    # Tracks which agents have completed — prevents re-running the same agent
    agents_called: Annotated[list[str], operator.add]

    # Structured findings from each specialist agent — appended, never replaced
    agent_findings: Annotated[list[dict], operator.add]


# ═══════════════════════════════════════════════════════════════════════════════
# Supervisor routing schema
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_AGENTS = Literal["merchant_intel", "customer_profile", "velocity_agent", "aggregate"]

class SupervisorDecision(BaseModel):
    """
    The supervisor's routing decision after each agent completes.
    Using structured output ensures the supervisor always returns a valid agent name.
    """
    next: AVAILABLE_AGENTS = Field(
        description=(
            "Which specialist agent to invoke next. "
            "Use 'aggregate' when all necessary agents have been called."
        )
    )
    reasoning: str = Field(description="One sentence explaining this routing choice.")


SUPERVISOR_SYSTEM_PROMPT = """\
You are a fraud investigation supervisor. Your job is to decide which specialist \
agent to call next, given what has already been investigated.

Available agents:
  merchant_intel    — Analyses merchant fraud history, prior disputes, skimmer devices
  customer_profile  — Analyses customer spend patterns, device fingerprint, location history
  velocity_agent    — Detects card-cloning via rapid sequential charges (only needed when
                      the merchant has a known skimmer OR multiple identical charges exist)
  aggregate         — Final step: combines all findings into a fraud score (call this last)

Rules:
  1. Always call merchant_intel first if not yet called.
  2. Always call customer_profile if not yet called.
  3. Call velocity_agent if merchant has a known_skimmer=True OR if multiple rapid charges
     are visible in the transaction data — AND velocity_agent has not yet been called.
  4. Call aggregate when merchant_intel and customer_profile have both been called
     (and velocity_agent has been called if warranted).
  5. Never call the same agent twice.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════════

def intake(state: SupervisorState) -> dict:
    """Load the transaction. Identical to Phase 1.2."""
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
        "next_agent": "",
        "agents_called": [],
        "agent_findings": [],
    }


async def supervisor(state: SupervisorState) -> dict:
    """
    Orchestrator node. Uses the cheap classifier model to decide which
    specialist agent runs next. Loops until it chooses 'aggregate'.

    This is the core of the supervisor pattern: a stateful LLM that tracks
    what has been done and decides what to do next.
    """
    agents_done = state.get("agents_called", [])
    findings_summary = "\n".join(
        f"  - {f['agent']}: {f['summary']}"
        for f in state.get("agent_findings", [])
    ) or "  (none yet)"

    txn = state.get("transaction", {})
    merchant = state.get("merchant_profile", {})

    context = f"""
Dispute: {state['dispute_id']}
Transaction: ${txn.get('amount', 0):.2f} at {txn.get('merchant', 'unknown')} ({txn.get('location', '')})
Merchant known_skimmer: {merchant.get('known_skimmer', 'unknown (merchant_intel not yet called)')}

Agents already called: {agents_done if agents_done else 'none'}
Findings so far:
{findings_summary}

Which agent should run next?
""".strip()

    llm = get_classifier(max_tokens=128)
    decision: SupervisorDecision = await llm.with_structured_output(SupervisorDecision).ainvoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    return {"next_agent": decision.next}


# ── Specialist agents ─────────────────────────────────────────────────────────
# These are deterministic — they fetch data and compute structured findings.
# No LLM calls. Each agent has a single, focused responsibility.

def merchant_intel(state: SupervisorState) -> dict:
    """
    Specialist: Merchant Intelligence
    Responsibility: merchant fraud history, prior disputes, skimmer devices.
    Knows nothing about the customer — that's CustomerProfileAgent's job.
    """
    txn = state["transaction"]
    profile = check_merchant_fraud_history.invoke({"merchant_id": txn["merchant_id"]})

    risk_flags = []
    fraud_signal = 0.0

    if profile.get("known_skimmer"):
        risk_flags.append("KNOWN CARD SKIMMER DEVICE")
        fraud_signal += 0.4
    if profile.get("prior_dispute_count", 0) > 20:
        risk_flags.append(f"{profile['prior_dispute_count']} prior disputes")
        fraud_signal += 0.3
    if profile.get("fraud_rate_pct", 0) > 15:
        risk_flags.append(f"{profile['fraud_rate_pct']}% fraud rate")
        fraud_signal += 0.2
    if not risk_flags:
        risk_flags.append(f"clean — {profile.get('prior_dispute_count',0)} disputes, {profile.get('fraud_rate_pct',0)}% fraud rate")
        fraud_signal -= 0.2

    summary = f"Merchant {profile.get('merchant_name')}: {'; '.join(risk_flags)}"

    return {
        "merchant_profile": profile,
        "agents_called": ["merchant_intel"],
        "agent_findings": [{
            "agent": "merchant_intel",
            "summary": summary,
            "fraud_signal": fraud_signal,
            "flags": risk_flags,
            "raw": profile,
        }],
        "evidence": [f"[MerchantIntel] {summary}"],
    }


def customer_profile(state: SupervisorState) -> dict:
    """
    Specialist: Customer Profile
    Responsibility: spend patterns, device fingerprint, location history.
    Knows nothing about merchant risk — that's MerchantIntelAgent's job.
    """
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
    recent_locs.add(home_city)
    recent_locs.add("Online")
    if txn.get("location") and txn["location"] not in recent_locs:
        risk_flags.append(f"location anomaly: {txn['location']} vs home {home_city}")
        fraud_signal += 0.2

    avg = profile.get("avg_transaction_amount", 0)
    if avg > 0 and txn["amount"] > avg * 3:
        risk_flags.append(f"amount {txn['amount']:.0f} is {txn['amount']/avg:.1f}× avg ({avg:.0f})")
        fraud_signal += 0.2

    merchant_name = state.get("merchant_profile", {}).get("merchant_name", "").lower()
    recurring = [
        t for t in profile.get("recent_transactions", [])
        if merchant_name and merchant_name in t.get("merchant", "").lower()
    ]
    if recurring:
        risk_flags.append(f"recurring: {len(recurring)} prior charge(s) at same merchant")
        fraud_signal -= 0.4

    summary = (
        f"Customer {state['customer_id']} ({home_city}): "
        + ("; ".join(risk_flags) if risk_flags else "no anomalies detected")
    )

    return {
        "customer_profile": profile,
        "agents_called": ["customer_profile"],
        "agent_findings": [{
            "agent": "customer_profile",
            "summary": summary,
            "fraud_signal": fraud_signal,
            "flags": risk_flags,
            "raw": profile,
        }],
        "evidence": [f"[CustomerProfile] {summary}"],
    }


def velocity_agent(state: SupervisorState) -> dict:
    """
    Specialist: Velocity Agent
    Responsibility: card-cloning detection via rapid sequential charges.
    Only invoked by the supervisor when warranted (skimmer present or rapid charges).
    This is the key differentiator for DISP-004.
    """
    result = check_velocity_pattern.invoke({"customer_id": state["customer_id"]})

    fraud_signal = 0.0
    flags = []

    if result.get("is_velocity_attack"):
        fraud_signal = 0.5
        flags.append(
            f"VELOCITY ATTACK: {result['charge_count']} charges of "
            f"${result['total_amount']/result['charge_count']:.2f} "
            f"in {result['time_span_minutes']} min (total ${result['total_amount']:.2f})"
        )
    else:
        flags.append("no velocity attack detected")

    summary = "; ".join(flags)

    return {
        "agents_called": ["velocity_agent"],
        "agent_findings": [{
            "agent": "velocity_agent",
            "summary": summary,
            "fraud_signal": fraud_signal,
            "flags": flags,
            "raw": result,
        }],
        "evidence": [f"[VelocityAgent] {summary}"],
    }


# ── Aggregate node ────────────────────────────────────────────────────────────

class AggregateOutput(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_factors: list[str]


AGGREGATE_SYSTEM_PROMPT = """\
You are a fraud analyst. You will receive structured findings from specialist agents.
Each finding includes a fraud_signal (positive = increases fraud probability,
negative = decreases it). Combine them into a single fraud_score (0.0–1.0).

Important: the final score should reflect the weight of evidence across all agents,
not just the arithmetic sum. Cap at 1.0, floor at 0.0.
"""


async def aggregate(state: SupervisorState) -> dict:
    """
    Final aggregation node. Receives all specialist findings and makes ONE LLM
    call to produce the composite fraud score. Same Gather-then-Reason pattern.
    """
    findings = state.get("agent_findings", [])
    txn = state["transaction"]

    findings_text = "\n".join(
        f"  {f['agent']} (signal: {f['fraud_signal']:+.1f}): {f['summary']}"
        for f in findings
    )

    prompt = f"""
Dispute: {state['dispute_id']}
Transaction: ${txn['amount']:.2f} at {txn['merchant']} ({txn['location']})

Specialist findings:
{findings_text}

Compute the composite fraud_score, list key_factors, and provide reasoning.
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


# ── Outcome nodes (same as Phase 1.2) ────────────────────────────────────────

def auto_credit(state: SupervisorState) -> dict:
    txn = state["transaction"]
    return {
        "decision": "auto_credit",
        "resolution_amount": txn["amount"],
        "evidence": [f"AUTO-CREDIT ${txn['amount']:.2f} — score {state['fraud_score']:.2f}"],
    }

def deny(state: SupervisorState) -> dict:
    return {
        "decision": "deny",
        "resolution_amount": 0.0,
        "evidence": [f"DENY — score {state['fraud_score']:.2f}, transaction appears legitimate"],
    }

def human_review(state: SupervisorState) -> dict:
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

def notify(state: SupervisorState) -> dict:
    decision = state["decision"]
    txn = state["transaction"]
    messages = {
        "auto_credit":  f"Dispute approved — ${state['resolution_amount']:.2f} credit issued for {txn['merchant']}.",
        "deny":         f"Dispute denied — ${txn['amount']:.2f} at {txn['merchant']} appears legitimate.",
        "human_review": f"Dispute under review — specialist will contact you within 2 business days.",
    }
    return {
        "notification_sent": True,
        "evidence": [f"Notification: {messages.get(decision, '')}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routing functions
# ═══════════════════════════════════════════════════════════════════════════════

def supervisor_route(state: SupervisorState) -> str:
    """
    Edge function: reads supervisor's next_agent decision and routes accordingly.
    The supervisor loop continues until next_agent == 'aggregate'.
    """
    return state.get("next_agent", "aggregate")


def decide_route(state: SupervisorState) -> Literal["auto_credit", "deny", "human_review"]:
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
    g = StateGraph(SupervisorState)

    g.add_node("intake",          intake)
    g.add_node("supervisor",      supervisor)
    g.add_node("merchant_intel",  merchant_intel)
    g.add_node("customer_profile",customer_profile)
    g.add_node("velocity_agent",  velocity_agent)
    g.add_node("aggregate",       aggregate)
    g.add_node("auto_credit",     auto_credit)
    g.add_node("deny",            deny)
    g.add_node("human_review",    human_review)
    g.add_node("notify",          notify)

    # Intake → supervisor (supervisor decides first agent)
    g.set_entry_point("intake")
    g.add_edge("intake", "supervisor")

    # Supervisor routes to one of: merchant_intel, customer_profile,
    # velocity_agent, or aggregate (done)
    g.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "merchant_intel":   "merchant_intel",
            "customer_profile": "customer_profile",
            "velocity_agent":   "velocity_agent",
            "aggregate":        "aggregate",
        },
    )

    # After each specialist, loop back to supervisor
    g.add_edge("merchant_intel",   "supervisor")
    g.add_edge("customer_profile", "supervisor")
    g.add_edge("velocity_agent",   "supervisor")

    # Aggregate → decide → outcome → notify
    g.add_conditional_edges(
        "aggregate", decide_route,
        {"auto_credit": "auto_credit", "deny": "deny", "human_review": "human_review"},
    )
    g.add_edge("auto_credit",   "notify")
    g.add_edge("deny",          "notify")
    g.add_edge("human_review",  "notify")
    g.add_edge("notify",        END)

    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "intake":           "blue",
    "supervisor":       "bright_white",
    "merchant_intel":   "cyan",
    "customer_profile": "cyan",
    "velocity_agent":   "magenta",
    "aggregate":        "yellow",
    "auto_credit":      "green",
    "deny":             "red",
    "human_review":     "yellow",
    "notify":           "white",
}

DECISION_COLORS = {
    "auto_credit": "green", "deny": "red",
    "human_review": "yellow", "pending": "dim",
}


async def run_dispute(app, dispute_id: str):
    dispute = DISPUTES[dispute_id]
    initial: SupervisorState = {
        "dispute_id": dispute_id,
        "customer_id": dispute["customer_id"],
        "transaction": {}, "merchant_profile": {}, "customer_profile": {},
        "evidence": [], "fraud_score": 0.0, "decision": "pending",
        "resolution_amount": 0.0, "human_notes": "", "analyst_approved": None,
        "notification_sent": False, "error": None,
        "next_agent": "", "agents_called": [], "agent_findings": [],
    }
    config = {"configurable": {"thread_id": f"2_1_{dispute_id}"}}

    console.print(f"\n[bold]Dispute {dispute_id}[/bold]  customer: {dispute['customer_id']}")
    console.print("─" * 70)

    supervisor_call_count = 0

    async for update in app.astream(initial, config=config, stream_mode="updates"):
        for node_name, changes in update.items():
            color = NODE_COLORS.get(node_name, "white")

            extras = []
            if node_name == "supervisor":
                supervisor_call_count += 1
                next_a = changes.get("next_agent", "?")
                extras.append(f"[dim]call #{supervisor_call_count}[/dim] → routing to [bold]{next_a}[/bold]")
            elif node_name == "intake" and changes.get("transaction"):
                txn = changes["transaction"]
                extras.append(f"${txn['amount']:.2f} at {txn['merchant']}")
            elif node_name == "aggregate" and "fraud_score" in changes:
                extras.append(f"composite score → [bold]{changes['fraud_score']:.2f}[/bold]")
            elif node_name in ("auto_credit", "deny", "human_review"):
                d = changes.get("decision", node_name)
                dc = DECISION_COLORS.get(d, "white")
                extras.append(f"[bold {dc}]{d.upper().replace('_',' ')}[/bold {dc}]")
            elif node_name == "notify":
                extras.append("notification sent ✓")

            extra_str = "  " + " | ".join(extras) if extras else ""
            console.print(
                f"  [{color}]■[/{color}] [bold {color}]{node_name:<18}[/bold {color}]{extra_str}"
            )

            for item in changes.get("evidence", []):
                console.print(f"               [dim]↳ {item}[/dim]")

    final = (await app.aget_state(config)).values
    _print_final(dispute_id, final, supervisor_call_count)


def _print_final(dispute_id: str, state: dict, supervisor_calls: int):
    decision = state.get("decision", "pending")
    color = DECISION_COLORS.get(decision, "white")

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("k", style="dim", width=22)
    table.add_column("v")
    table.add_row("Fraud score",      f"[bold]{state.get('fraud_score', 0):.2f}[/bold]")
    table.add_row("Decision",         f"[bold {color}]{decision.upper().replace('_',' ')}[/bold {color}]")
    amt = state.get("resolution_amount", 0)
    table.add_row("Resolution",       f"${amt:.2f}" if amt > 0 else "—")
    table.add_row("Agents called",    str(state.get("agents_called", [])))
    table.add_row("Supervisor calls", str(supervisor_calls))
    table.add_row("Evidence items",   str(len(state.get("evidence", []))))

    console.print(Panel(table, title=f"[bold]{dispute_id} Final State[/bold]", border_style=color))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispute", choices=list(DISPUTES.keys()))
    args = parser.parse_args()

    cfg = current_config()
    console.print(Panel.fit(
        f"[bold cyan]Phase 2.1 — Supervisor Pattern[/bold cyan]\n"
        f"Supervisor orchestrates: MerchantIntel → CustomerProfile → [VelocityAgent?] → Aggregate\n"
        f"[dim]provider: {cfg['provider']}  "
        f"supervisor: {cfg['classifier']}  investigator: {cfg['investigator']}[/dim]",
        border_style="cyan",
    ))

    async with AsyncSqliteSaver.from_conn_string("disputes.db") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        try:
            png_path = "phase2_1_supervisor/graph.png"
            app.get_graph().draw_mermaid_png(output_file_path=png_path)
            console.print(f"[dim]Graph saved → {png_path}[/dim]\n")
        except Exception:
            console.print("[dim]Graph PNG skipped (mermaid.ink unreachable — paste the Mermaid output into mermaid.live)[/dim]\n")

        dispute_ids = [args.dispute] if args.dispute else list(DISPUTES.keys())
        for dispute_id in dispute_ids:
            await run_dispute(app, dispute_id)

        console.print(Rule("[dim]All disputes complete[/dim]"))


if __name__ == "__main__":
    asyncio.run(main())
