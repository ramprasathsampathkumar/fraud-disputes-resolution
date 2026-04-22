"""
Phase 3.2 — Human-in-the-Loop

Concepts demonstrated:
  - interrupt() pauses graph execution inside a node, saves state to SQLite,
    and waits for analyst input before continuing
  - Command(resume=value) injects the analyst's decision and resumes the graph
    from exactly where it paused — no state is lost
  - Two interrupt points with different triggers:
      issue_credit   — amount >= $500 (irreversible high-value action)
      final_decision — 0.4 < score < 0.7 (ambiguous evidence, needs judgment)
  - Structured evidence brief generated at each interrupt so the analyst has
    everything they need in one place — no digging through logs

Graph structure (extends Phase 3.1 with interrupt nodes):

    intake → memory_lookup → fan_out ──→ [parallel checks] → aggregate
      → decide_route
          ├── auto_credit   (score > 0.8, amount < $500 — no interrupt)
          ├── deny          (score < 0.4 — no interrupt)
          ├── issue_credit  (amount >= $500 — INTERRUPT: analyst must approve)
          └── final_decision(0.4 < score < 0.7 — INTERRUPT: ambiguous evidence)
              ↓
          store_outcome → notify

Routing for all 4 disputes:
  DISP-001 cold  ($340, score ~0.9)  → auto_credit       (no interrupt)
  DISP-001 warm  ($340, score ~0.55) → final_decision     (INTERRUPT — ambiguous after memory)
  DISP-002       ($1200, score ~0.6) → issue_credit       (INTERRUPT — high value)
  DISP-003       ($18, score ~0.05)  → deny               (no interrupt)
  DISP-004       ($340, score ~1.0)  → auto_credit        (no interrupt)

The "aha" moment — DISP-002:
  System investigates fully, produces structured brief, then pauses.
  Analyst approves or denies in seconds. Shows AI doing 90% of the work.

Run:
    # Interactive — graph pauses, prompts analyst in terminal, resumes automatically
    uv run python -m phase3_2_human_in_loop.graph --dispute DISP-002

    # Warm run — DISP-001 hits ambiguous score interrupt after memory context
    uv run python -m phase3_2_human_in_loop.graph --dispute DISP-001 --warm

    # Non-interactive — auto-approve all interrupts (useful for demos/testing)
    uv run python -m phase3_2_human_in_loop.graph --dispute DISP-002 --auto-approve

    # Auto-deny
    uv run python -m phase3_2_human_in_loop.graph --dispute DISP-002 --auto-deny
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
from rich.prompt import Prompt
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
from langgraph.types import Command, Send, interrupt

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

# Set by CLI — controls whether interrupts auto-resolve or prompt the analyst
AUTO_APPROVE: bool | None = None  # None = interactive, True = approve, False = deny


# ═══════════════════════════════════════════════════════════════════════════════
# State schema
# ═══════════════════════════════════════════════════════════════════════════════

class HumanLoopState(TypedDict):
    dispute_id: str
    customer_id: str
    transaction: dict
    merchant_profile: dict
    customer_profile: dict
    evidence: Annotated[list[str], operator.add]
    agent_findings: Annotated[list[dict], operator.add]
    memory_context: Annotated[list[str], operator.add]
    fraud_score: float
    decision: Literal["auto_credit", "deny", "human_review", "pending"]
    resolution_amount: float
    model_reasoning: str   # LLM aggregate reasoning — set by aggregate node
    human_notes: str       # Analyst notes — set only at interrupt nodes
    analyst_approved: bool | None
    analyst_decision: str | None   # "approve_credit" | "deny" | "request_more_info"
    notification_sent: bool
    error: str | None


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence brief generator
# Called inside both interrupt nodes so the analyst has a full picture.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_evidence_brief(state: HumanLoopState, interrupt_reason: str) -> dict:
    """
    Build the structured evidence brief handed to the analyst at interrupt.
    This is what transforms raw agent output into an actionable human brief.
    """
    txn = state["transaction"]
    findings = state.get("agent_findings", [])
    memory = state.get("memory_context", [])

    # Format check signals with visual indicators
    evidence_lines = []
    for f in findings:
        for flag in f.get("flags", []):
            signal = f.get("fraud_signal", 0)
            icon = "⚠" if signal > 0 else "✓"
            evidence_lines.append(f"  {icon} [{f['agent']}] {flag}")

    # Add memory context
    for mem in memory:
        evidence_lines.append(f"  📋 {mem}")

    score = state.get("fraud_score", 0)
    score_label = (
        "HIGH RISK" if score > 0.7
        else "AMBIGUOUS" if score > 0.4
        else "LOW RISK"
    )

    return {
        "dispute_id": state["dispute_id"],
        "merchant": txn.get("merchant", ""),
        "amount": txn.get("amount", 0),
        "location": txn.get("location", ""),
        "customer_id": state["customer_id"],
        "fraud_score": score,
        "score_label": score_label,
        "interrupt_reason": interrupt_reason,
        "evidence_lines": evidence_lines,
        "model_reasoning": state.get("model_reasoning", ""),
        "actions": ["approve_credit", "deny", "request_more_info"],
    }


def print_evidence_brief(brief: dict):
    """Render the evidence brief as a rich panel in the terminal."""
    txn_line = (
        f"[bold]DISPUTE:[/bold] {brief['dispute_id']} | "
        f"[bold]Amount:[/bold] ${brief['amount']:.2f} | "
        f"[bold]Merchant:[/bold] {brief['merchant']} ({brief['location']})"
    )

    score = brief["fraud_score"]
    score_color = "red" if score > 0.7 else "yellow" if score > 0.4 else "green"
    score_line = (
        f"[bold]Fraud Score:[/bold] "
        f"[bold {score_color}]{score:.2f} ({brief['score_label']})[/bold {score_color}]"
    )

    evidence_text = "\n".join(brief["evidence_lines"]) or "  (no evidence collected)"

    reasoning = brief.get("model_reasoning", "")
    reasoning_section = f"\n[dim]Model reasoning: {reasoning[:200]}[/dim]" if reasoning else ""

    actions = " | ".join(f"[bold]{a}[/bold]" for a in brief["actions"])

    content = (
        f"{txn_line}\n"
        f"Customer: {brief['customer_id']}\n\n"
        f"[bold]EVIDENCE:[/bold]\n{evidence_text}"
        f"{reasoning_section}\n\n"
        f"{score_line}\n"
        f"[dim]Trigger: {brief['interrupt_reason']}[/dim]\n\n"
        f"[bold]ACTION REQUIRED:[/bold] {actions}"
    )

    console.print(Panel(
        content,
        title="[bold yellow]⏸  ANALYST REVIEW REQUIRED[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))


def get_analyst_decision(brief: dict) -> dict:
    """
    Prompt analyst for a decision (interactive) or auto-resolve (demo mode).
    Returns the resume payload that Command(resume=...) will inject.
    """
    if AUTO_APPROVE is True:
        decision = "approve_credit"
        console.print(f"[dim][auto-approve] → {decision}[/dim]")
    elif AUTO_APPROVE is False:
        decision = "deny"
        console.print(f"[dim][auto-deny] → {decision}[/dim]")
    else:
        # Interactive — prompt analyst in terminal
        decision = Prompt.ask(
            "\nYour decision",
            choices=["approve_credit", "deny", "request_more_info"],
            default="approve_credit",
        )

    notes = ""
    if decision == "request_more_info" and AUTO_APPROVE is None:
        notes = Prompt.ask("Additional context (optional)", default="")

    return {"decision": decision, "analyst_notes": notes}


# ═══════════════════════════════════════════════════════════════════════════════
# Intake + memory lookup + fan-out (same as Phase 3.1)
# ═══════════════════════════════════════════════════════════════════════════════

def intake(state: HumanLoopState) -> dict:
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
        "analyst_approved": None,
        "analyst_decision": None,
    }


def memory_lookup(state: HumanLoopState) -> dict:
    txn = state["transaction"]
    merchant_name = txn.get("merchant", "")
    merchant_id = txn.get("merchant_id", "")
    customer_id = state["customer_id"]
    memories: list[str] = []

    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(COLLECTION_NAME)
        total = collection.count()

        if total > 0:
            n = max(1, min(3, total))
            r1 = collection.query(
                query_texts=[f"merchant {merchant_name} {merchant_id} fraud history"],
                n_results=n,
            )
            for doc in (r1["documents"] or [[]])[0]:
                memories.append(f"[Merchant memory] {doc}")

            r2 = collection.query(
                query_texts=[f"customer {customer_id} travel spending behaviour"],
                n_results=n,
            )
            for doc in (r2["documents"] or [[]])[0]:
                if doc not in memories:
                    memories.append(f"[Customer memory] {doc}")
    except Exception as e:
        memories.append(f"[Memory unavailable: {e}]")

    return {
        "memory_context": memories,
        "evidence": [
            f"Memory: {len(memories)} item(s) retrieved"
            if memories else "Memory: no prior knowledge"
        ],
    }


def fan_out(state: HumanLoopState) -> list[Send]:
    return [
        Send("merchant_check", state),
        Send("customer_check", state),
        Send("velocity_check", state),
    ]


def merchant_check(state: HumanLoopState) -> dict:
    txn = state["transaction"]
    profile = check_merchant_fraud_history.invoke({"merchant_id": txn["merchant_id"]})
    risk_flags, fraud_signal = [], 0.0
    if profile.get("known_skimmer"):
        risk_flags.append("KNOWN CARD SKIMMER"); fraud_signal += 0.4
    if profile.get("prior_dispute_count", 0) > 20:
        risk_flags.append(f"{profile['prior_dispute_count']} prior disputes"); fraud_signal += 0.3
    if profile.get("fraud_rate_pct", 0) > 15:
        risk_flags.append(f"{profile['fraud_rate_pct']}% fraud rate"); fraud_signal += 0.2
    if not risk_flags:
        risk_flags.append(f"clean — {profile.get('prior_dispute_count',0)} disputes"); fraud_signal -= 0.2
    summary = f"Merchant {profile.get('merchant_name')}: {'; '.join(risk_flags)}"
    return {
        "merchant_profile": profile,
        "agent_findings": [{"agent": "merchant_check", "summary": summary, "fraud_signal": fraud_signal, "flags": risk_flags}],
        "evidence": [f"[MerchantCheck] {summary}"],
    }


def customer_check(state: HumanLoopState) -> dict:
    txn = state["transaction"]
    profile = get_customer_spend_history.invoke({"customer_id": state["customer_id"]})
    risk_flags, fraud_signal = [], 0.0
    known_devices = profile.get("known_devices", [])
    if txn.get("device_id") and txn["device_id"] not in known_devices:
        risk_flags.append(f"unrecognised device {txn['device_id']}"); fraud_signal += 0.2
    home_city = profile.get("home_city", "")
    recent_locs = {t.get("location","") for t in profile.get("recent_transactions",[])}
    recent_locs.update([home_city, "Online"])
    if txn.get("location") and txn["location"] not in recent_locs:
        risk_flags.append(f"location anomaly: {txn['location']} vs home {home_city}"); fraud_signal += 0.2
    avg = profile.get("avg_transaction_amount", 0)
    if avg > 0 and txn["amount"] > avg * 3:
        risk_flags.append(f"amount ${txn['amount']:.0f} is {txn['amount']/avg:.1f}× avg"); fraud_signal += 0.2
    recurring = [t for t in profile.get("recent_transactions",[]) if txn.get("merchant","").lower() in t.get("merchant","").lower()]
    if recurring:
        risk_flags.append(f"recurring: {len(recurring)} prior charge(s)"); fraud_signal -= 0.4
    if not risk_flags:
        risk_flags.append("no anomalies detected")
    summary = f"Customer {state['customer_id']} ({home_city}): {'; '.join(risk_flags)}"
    return {
        "customer_profile": profile,
        "agent_findings": [{"agent": "customer_check", "summary": summary, "fraud_signal": fraud_signal, "flags": risk_flags}],
        "evidence": [f"[CustomerCheck] {summary}"],
    }


def velocity_check(state: HumanLoopState) -> dict:
    result = check_velocity_pattern.invoke({"customer_id": state["customer_id"]})
    fraud_signal, flags = 0.0, []
    if result.get("is_velocity_attack"):
        fraud_signal = 0.5
        flags.append(f"VELOCITY ATTACK: {result['charge_count']} charges in {result['time_span_minutes']} min")
    else:
        flags.append("no velocity attack detected")
    summary = "; ".join(flags)
    return {
        "agent_findings": [{"agent": "velocity_check", "summary": summary, "fraud_signal": fraud_signal, "flags": flags}],
        "evidence": [f"[VelocityCheck] {summary}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate (memory-aware, same as Phase 3.1)
# ═══════════════════════════════════════════════════════════════════════════════

class AggregateOutput(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_factors: list[str]

AGGREGATE_SYSTEM_PROMPT = """\
You are a senior fraud analyst. Combine parallel check findings and any prior \
memory context into a single fraud_score (0.0–1.0). Memory context takes \
precedence — known legitimate patterns reduce anomaly weight.
"""

async def aggregate(state: HumanLoopState) -> dict:
    findings = state.get("agent_findings", [])
    memory = state.get("memory_context", [])
    txn = state["transaction"]

    findings_text = "\n".join(
        f"  {f['agent']} (signal: {f['fraud_signal']:+.1f}): {f['summary']}"
        for f in findings
    )
    memory_text = (
        "\n".join(f"  - {m}" for m in memory)
        if memory else "  (none)"
    )

    prompt = f"""
Dispute: {state['dispute_id']}
Transaction: ${txn['amount']:.2f} at {txn['merchant']} ({txn['location']})

Parallel check results:
{findings_text}

Prior knowledge from long-term memory:
{memory_text}

Compute fraud_score, key_factors, and reasoning.
""".strip()

    llm = get_investigator(max_tokens=512)
    result: AggregateOutput = await llm.with_structured_output(AggregateOutput).ainvoke([
        SystemMessage(content=AGGREGATE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return {
        "fraud_score": result.fraud_score,
        "model_reasoning": result.reasoning,
        "evidence": result.key_factors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Interrupt nodes — Phase 3.2's core addition
# ═══════════════════════════════════════════════════════════════════════════════

def issue_credit(state: HumanLoopState) -> dict:
    """
    Interrupt point 1 — High-value credit (amount >= $500).

    Pauses graph execution, generates a structured evidence brief, and waits
    for analyst approval before issuing a potentially large credit.

    The interrupt() call:
      1. Serialises state to SQLite checkpointer
      2. Yields the brief payload to the caller (CLI or Studio)
      3. Blocks until Command(resume=analyst_decision) is received
      4. Returns analyst_decision — execution continues from here
    """
    txn = state["transaction"]
    brief = generate_evidence_brief(
        state,
        interrupt_reason=f"Amount ${txn['amount']:.2f} ≥ $500 threshold — senior analyst approval required",
    )

    # ── PAUSE ──────────────────────────────────────────────────────────────────
    analyst_input: dict = interrupt(brief)
    # ── RESUME (analyst_input injected by Command(resume=...)) ─────────────────

    decision = analyst_input.get("decision", "deny")
    notes = analyst_input.get("analyst_notes", "")

    if decision == "approve_credit":
        return {
            "decision": "auto_credit",
            "resolution_amount": txn["amount"],
            "analyst_approved": True,
            "analyst_decision": decision,
            "human_notes": notes or state.get("model_reasoning", ""),
            "evidence": [
                f"ANALYST APPROVED: ${txn['amount']:.2f} credit authorised by human reviewer"
            ],
        }
    elif decision == "request_more_info":
        return {
            "decision": "human_review",
            "resolution_amount": txn["amount"],
            "analyst_approved": None,
            "analyst_decision": decision,
            "human_notes": notes,
            "evidence": [f"ANALYST: additional information requested — {notes or 'no details provided'}"],
        }
    else:
        return {
            "decision": "deny",
            "resolution_amount": 0.0,
            "analyst_approved": False,
            "analyst_decision": decision,
            "human_notes": notes,
            "evidence": ["ANALYST DENIED: dispute rejected after human review"],
        }


def final_decision(state: HumanLoopState) -> dict:
    """
    Interrupt point 2 — Ambiguous score (0.4 < score < 0.7).

    The evidence is genuinely conflicting — the system cannot confidently decide.
    Pauses to hand a clean brief to a human investigator who can apply judgment
    that the model cannot (e.g. knowing the customer personally called in,
    or recognising a known fraud ring pattern not yet in memory).
    """
    score = state["fraud_score"]
    brief = generate_evidence_brief(
        state,
        interrupt_reason=f"Ambiguous fraud score {score:.2f} — conflicting signals require human judgment",
    )

    # ── PAUSE ──────────────────────────────────────────────────────────────────
    analyst_input: dict = interrupt(brief)
    # ── RESUME ─────────────────────────────────────────────────────────────────

    decision = analyst_input.get("decision", "deny")
    notes = analyst_input.get("analyst_notes", "")
    txn = state["transaction"]

    if decision == "approve_credit":
        return {
            "decision": "auto_credit",
            "resolution_amount": txn["amount"],
            "analyst_approved": True,
            "analyst_decision": decision,
            "human_notes": notes,
            "evidence": [f"ANALYST RULED FRAUD: credit of ${txn['amount']:.2f} approved"],
        }
    elif decision == "request_more_info":
        return {
            "decision": "human_review",
            "resolution_amount": txn["amount"],
            "analyst_approved": None,
            "analyst_decision": decision,
            "human_notes": notes,
            "evidence": [f"ANALYST: escalated for further investigation — {notes or 'pending'}"],
        }
    else:
        return {
            "decision": "deny",
            "resolution_amount": 0.0,
            "analyst_approved": False,
            "analyst_decision": decision,
            "human_notes": notes,
            "evidence": ["ANALYST RULED LEGITIMATE: dispute denied after review"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Non-interrupt outcome nodes (clear-cut cases — no human needed)
# ═══════════════════════════════════════════════════════════════════════════════

def auto_credit(state: HumanLoopState) -> dict:
    txn = state["transaction"]
    return {
        "decision": "auto_credit",
        "resolution_amount": txn["amount"],
        "analyst_approved": False,  # system decision, no analyst involved
        "evidence": [f"AUTO-CREDIT ${txn['amount']:.2f} — score {state['fraud_score']:.2f} (system decision)"],
    }

def deny(state: HumanLoopState) -> dict:
    return {
        "decision": "deny",
        "resolution_amount": 0.0,
        "analyst_approved": False,
        "evidence": [f"AUTO-DENY — score {state['fraud_score']:.2f}, transaction appears legitimate"],
    }


def store_outcome(state: HumanLoopState) -> dict:
    txn = state["transaction"]
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(COLLECTION_NAME)
        dispute_id = state["dispute_id"]
        decision = state["decision"]
        score = state["fraud_score"]
        analyst_note = (
            f" Analyst decision: {state.get('analyst_decision','N/A')}."
            if state.get("analyst_decision") else ""
        )
        doc = (
            f"Dispute {dispute_id}: ${txn['amount']:.2f} at {txn['merchant']} "
            f"({txn['location']}) resolved as {decision.upper()} "
            f"(score: {score:.2f}).{analyst_note}"
        )
        collection.upsert(
            ids=[f"{dispute_id}_3_2"],
            documents=[doc],
            metadatas=[{
                "dispute_id": dispute_id,
                "merchant_id": txn.get("merchant_id", ""),
                "customer_id": state["customer_id"],
                "decision": decision,
                "fraud_score": score,
                "analyst_involved": bool(state.get("analyst_decision")),
                "type": "dispute_outcome",
            }],
        )
    except Exception:
        pass
    return {"evidence": ["Outcome stored to ChromaDB"]}


def notify(state: HumanLoopState) -> dict:
    decision = state["decision"]
    txn = state["transaction"]
    analyst_tag = " (analyst reviewed)" if state.get("analyst_decision") else ""
    messages = {
        "auto_credit":  f"Dispute approved — ${state['resolution_amount']:.2f} credit issued for {txn['merchant']}{analyst_tag}.",
        "deny":         f"Dispute denied — ${txn['amount']:.2f} at {txn['merchant']} appears legitimate{analyst_tag}.",
        "human_review": f"Dispute under further investigation — you will be contacted within 1 business day.",
    }
    return {
        "notification_sent": True,
        "evidence": [f"Notification: {messages.get(decision, '')}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════════════

def decide_route(state: HumanLoopState) -> Literal["auto_credit", "deny", "issue_credit", "final_decision"]:
    """
    Deterministic routing after aggregate.

    Four paths:
      auto_credit   — high confidence fraud, small amount (safe to automate)
      deny          — clearly legitimate (safe to automate)
      issue_credit  — high value: always needs analyst eyes, regardless of score
      final_decision— ambiguous score: system isn't confident enough to decide
    """
    score = state["fraud_score"]
    amount = state["transaction"]["amount"]

    if score > 0.8 and amount < 500:
        return "auto_credit"
    elif score < 0.4:
        return "deny"
    elif amount >= 500:
        return "issue_credit"     # interrupt point 1
    else:
        return "final_decision"   # interrupt point 2


# ═══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(checkpointer=None):
    g = StateGraph(HumanLoopState)

    g.add_node("intake",          intake)
    g.add_node("memory_lookup",   memory_lookup)
    g.add_node("merchant_check",  merchant_check)
    g.add_node("customer_check",  customer_check)
    g.add_node("velocity_check",  velocity_check)
    g.add_node("aggregate",       aggregate)
    g.add_node("auto_credit",     auto_credit)
    g.add_node("deny",            deny)
    g.add_node("issue_credit",    issue_credit)    # interrupt node 1
    g.add_node("final_decision",  final_decision)  # interrupt node 2
    g.add_node("store_outcome",   store_outcome)
    g.add_node("notify",          notify)

    g.set_entry_point("intake")
    g.add_edge("intake", "memory_lookup")
    g.add_conditional_edges("memory_lookup", fan_out)
    g.add_edge("merchant_check", "aggregate")
    g.add_edge("customer_check", "aggregate")
    g.add_edge("velocity_check", "aggregate")

    g.add_conditional_edges(
        "aggregate",
        decide_route,
        {
            "auto_credit":    "auto_credit",
            "deny":           "deny",
            "issue_credit":   "issue_credit",
            "final_decision": "final_decision",
        },
    )

    # All outcome paths converge at store_outcome → notify → END
    for outcome in ("auto_credit", "deny", "issue_credit", "final_decision"):
        g.add_edge(outcome, "store_outcome")
    g.add_edge("store_outcome", "notify")
    g.add_edge("notify", END)

    if not isinstance(checkpointer, BaseCheckpointSaver):
        checkpointer = None
    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "intake":          "blue",
    "memory_lookup":   "bright_magenta",
    "merchant_check":  "cyan",
    "customer_check":  "cyan",
    "velocity_check":  "magenta",
    "aggregate":       "yellow",
    "auto_credit":     "green",
    "deny":            "red",
    "issue_credit":    "bold yellow",
    "final_decision":  "bold yellow",
    "store_outcome":   "bright_magenta",
    "notify":          "white",
}
DECISION_COLORS = {
    "auto_credit": "green", "deny": "red",
    "human_review": "yellow", "pending": "dim",
}


async def run_dispute(app, dispute_id: str):
    dispute = DISPUTES[dispute_id]
    initial: HumanLoopState = {
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
        "model_reasoning":  "",
        "human_notes":      "",
        "analyst_approved": None,
        "analyst_decision": None,
        "notification_sent":False,
        "error":            None,
    }
    config = {"configurable": {"thread_id": f"3_2_{dispute_id}"}}

    console.print(f"\n[bold]Dispute {dispute_id}[/bold]  customer: {dispute['customer_id']}")
    console.print("─" * 70)

    t0 = time.perf_counter()
    interrupted = False
    interrupt_brief = None

    # ── Phase 1: run until interrupt (or completion) ──────────────────────────
    async for update in app.astream(initial, config=config, stream_mode="updates"):
        for node_name, changes in update.items():

            if node_name == "__interrupt__":
                # Graph paused — extract the brief from the interrupt payload
                interrupted = True
                interrupt_brief = changes[0].value if changes else {}
                break

            color = NODE_COLORS.get(node_name, "white")
            extras = []

            if node_name == "intake":
                txn = changes.get("transaction", {})
                if txn:
                    extras.append(f"${txn['amount']:.2f} at {txn['merchant']}")
            elif node_name == "memory_lookup":
                mems = changes.get("memory_context", [])
                extras.append(f"[bright_magenta]{len(mems)} memory item(s)[/bright_magenta]" if mems else "[dim]no prior knowledge[/dim]")
            elif node_name == "aggregate" and "fraud_score" in changes:
                score = changes["fraud_score"]
                sc = "green" if score < 0.4 else ("red" if score > 0.8 else "yellow")
                extras.append(f"score → [bold {sc}]{score:.2f}[/bold {sc}]")
            elif node_name in ("auto_credit", "deny"):
                d = changes.get("decision", node_name)
                dc = DECISION_COLORS.get(d, "white")
                extras.append(f"[bold {dc}]{d.upper().replace('_',' ')}[/bold {dc}] (system decision)")
            elif node_name == "store_outcome":
                extras.append("stored to ChromaDB ✓")
            elif node_name == "notify":
                extras.append("notification sent ✓")

            console.print(
                f"  [{color}]■[/{color}] [bold {color}]{node_name:<18}[/bold {color}]"
                + (f"  {'  '.join(extras)}" if extras else "")
            )
            for item in changes.get("evidence", []):
                console.print(f"               [dim]↳ {item}[/dim]")
            for mem in changes.get("memory_context", []):
                console.print(f"               [dim bright_magenta]↳ {mem[:100]}[/dim bright_magenta]")

    # ── Phase 2: handle interrupt ─────────────────────────────────────────────
    if interrupted and interrupt_brief:
        console.print()
        print_evidence_brief(interrupt_brief)
        analyst_input = get_analyst_decision(interrupt_brief)

        console.print(f"\n[dim]Resuming graph with analyst decision: [bold]{analyst_input['decision']}[/bold][/dim]\n")

        # Resume with Command(resume=analyst_input)
        async for update in app.astream(
            Command(resume=analyst_input),
            config=config,
            stream_mode="updates",
        ):
            for node_name, changes in update.items():
                if node_name == "__interrupt__":
                    continue
                color = NODE_COLORS.get(node_name, "white")
                extras = []

                if node_name in ("issue_credit", "final_decision"):
                    d = changes.get("decision", "")
                    dc = DECISION_COLORS.get(d, "yellow")
                    extras.append(f"[bold {dc}]{d.upper().replace('_',' ')}[/bold {dc}] (analyst: {changes.get('analyst_decision','')})")
                elif node_name == "store_outcome":
                    extras.append("stored to ChromaDB ✓")
                elif node_name == "notify":
                    extras.append("notification sent ✓")

                console.print(
                    f"  [{color}]■[/{color}] [bold {color}]{node_name:<18}[/bold {color}]"
                    + (f"  {'  '.join(extras)}" if extras else "")
                )
                for item in changes.get("evidence", []):
                    console.print(f"               [dim]↳ {item}[/dim]")

    total_elapsed = time.perf_counter() - t0
    final = (await app.aget_state(config)).values
    _print_final(dispute_id, final, total_elapsed, interrupted)


def _print_final(dispute_id: str, state: dict, elapsed: float, interrupted: bool):
    decision = state.get("decision", "pending")
    color = DECISION_COLORS.get(decision, "white")

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("k", style="dim", width=22)
    table.add_column("v")
    table.add_row("Fraud score",      f"[bold]{state.get('fraud_score', 0):.2f}[/bold]")
    table.add_row("Decision",         f"[bold {color}]{decision.upper().replace('_',' ')}[/bold {color}]")
    amt = state.get("resolution_amount", 0)
    table.add_row("Resolution",       f"${amt:.2f}" if amt > 0 else "—")
    table.add_row("Analyst involved", "Yes — " + str(state.get("analyst_decision","")) if interrupted else "No (system decision)")
    table.add_row("Memory items",     str(len(state.get("memory_context", []))))
    table.add_row("Evidence items",   str(len(state.get("evidence", []))))
    table.add_row("Total elapsed",    f"{elapsed:.2f}s")

    console.print(Panel(table, title=f"[bold]{dispute_id} Final State[/bold]", border_style=color))


async def main():
    global AUTO_APPROVE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dispute", choices=list(DISPUTES.keys()))
    parser.add_argument("--warm",        action="store_true", help="Keep ChromaDB memory (don't clear)")
    parser.add_argument("--auto-approve",action="store_true", help="Auto-approve all analyst interrupts")
    parser.add_argument("--auto-deny",   action="store_true", help="Auto-deny all analyst interrupts")
    args = parser.parse_args()

    if args.auto_approve:
        AUTO_APPROVE = True
    elif args.auto_deny:
        AUTO_APPROVE = False

    cfg = current_config()
    console.print(Panel.fit(
        f"[bold cyan]Phase 3.2 — Human-in-the-Loop[/bold cyan]\n"
        f"interrupt() pauses graph · analyst reviews brief · Command(resume=) continues\n"
        f"Interrupt points: [yellow]issue_credit[/yellow] (amount ≥ $500) · "
        f"[yellow]final_decision[/yellow] (score 0.4–0.7)\n"
        f"[dim]provider: {cfg['provider']}  investigator: {cfg['investigator']}[/dim]",
        border_style="cyan",
    ))

    async with AsyncSqliteSaver.from_conn_string("disputes.db") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        try:
            app.get_graph().draw_mermaid_png(output_file_path="phase3_2_human_in_loop/graph.png")
            console.print("[dim]Graph saved → phase3_2_human_in_loop/graph.png[/dim]\n")
        except Exception:
            console.print("[dim]Graph PNG skipped[/dim]\n")

        dispute_ids = [args.dispute] if args.dispute else list(DISPUTES.keys())
        for dispute_id in dispute_ids:
            await run_dispute(app, dispute_id)

        console.print(Rule("[dim]All disputes complete[/dim]"))


if __name__ == "__main__":
    asyncio.run(main())
