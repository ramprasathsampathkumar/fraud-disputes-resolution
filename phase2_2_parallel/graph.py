"""
Phase 2.2 — Parallel Fan-out (Send API)

Concepts demonstrated:
  - Deterministic fan_out node replaces the supervisor LLM routing loop
  - langgraph.types.Send dispatches all 3 checks simultaneously
  - Annotated[list, operator.add] reducers safely merge parallel writes
  - 0 LLM routing calls (vs 3–4 in Phase 2.1) + faster wall-clock time
  - Map-reduce pattern: fan_out → [parallel checks] → aggregate

Graph structure:

    intake ──→ fan_out ──┬──→ merchant_check ──┐
                         ├──→ customer_check ──┼──→ aggregate → [decide] → outcome → notify
                         └──→ velocity_check ──┘

Phase 2.1 vs 2.2 comparison (for DISP-001):
  Phase 2.1 — supervisor calls agents one by one
    LLM calls:   4  (3 routing + 1 aggregate)
    Wall time:   ~4.5s sequential

  Phase 2.2 — Send API fans out in parallel
    LLM calls:   1  (aggregate only)
    Wall time:   ~2.0s (longest branch wins)

Run:
    uv run python -m phase2_2_parallel.graph
    uv run python -m phase2_2_parallel.graph --dispute DISP-004
    uv run python -m phase2_2_parallel.graph --no-simulate-latency
"""

import argparse
import asyncio
import logging
import operator
import os
import time
import warnings
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from typing_extensions import TypedDict

load_dotenv()

warnings.filterwarnings("ignore", message="create_react_agent has been moved", category=DeprecationWarning)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)

from langchain_core.messages import HumanMessage, SystemMessage
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

# Simulated per-check latencies (seconds) — mirrors real service call profiles:
#   customer_check: slowest (profile service aggregates 90-day history)
#   merchant_check: medium (fraud DB lookup)
#   velocity_check: fastest (analytics query on recent window)
# Toggle off with --no-simulate-latency to see raw mock-data speed.
SIMULATED_LATENCY: dict[str, float] = {
    "merchant_check": 1.5,
    "customer_check": 2.0,
    "velocity_check": 1.0,
}
SIMULATE_LATENCY = True  # overridden by CLI flag


# ═══════════════════════════════════════════════════════════════════════════════
# State schema
#
# Annotated[list, operator.add] is the critical design decision for parallel
# safety. When two branches run simultaneously and both return {"evidence": [...]},
# LangGraph applies the reducer — operator.add — to concatenate the lists instead
# of letting one branch overwrite the other.
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelState(TypedDict):
    # ── Core dispute fields ──────────────────────────────────────────────────
    dispute_id: str
    customer_id: str
    transaction: dict
    merchant_profile: dict
    customer_profile: dict

    # Accumulated across all parallel branches — reducer makes this thread-safe
    evidence: Annotated[list[str], operator.add]

    # Structured findings per check — each branch appends its own dict
    agent_findings: Annotated[list[dict], operator.add]

    # Scoring
    fraud_score: float

    # Decision lifecycle
    decision: Literal["auto_credit", "deny", "human_review", "pending"]
    resolution_amount: float
    human_notes: str
    analyst_approved: bool | None
    notification_sent: bool
    error: str | None


# ═══════════════════════════════════════════════════════════════════════════════
# Intake node
# ═══════════════════════════════════════════════════════════════════════════════

def intake(state: ParallelState) -> dict:
    """Fetch the transaction. Identical to Phase 1.2 and 2.1."""
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
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-out — the core of Phase 2.2
#
# This function replaces the entire supervisor LLM loop from Phase 2.1.
# Instead of asking an LLM "what should I do next?", we deterministically
# return a list of Send objects — one per check. LangGraph executes them all
# concurrently and waits for all branches to complete before continuing.
#
# Cost comparison:
#   Phase 2.1:  3-4 LLM calls (routing) + 1 (aggregate) = 4-5 total
#   Phase 2.2:  0 LLM calls (routing) + 1 (aggregate)   = 1 total
# ═══════════════════════════════════════════════════════════════════════════════

def fan_out(state: ParallelState) -> list[Send]:
    """
    Deterministic fan-out: dispatch all 3 checks simultaneously.
    Each Send("node", state) creates an independent parallel branch.
    LangGraph merges their results using the Annotated reducers on state.
    """
    return [
        Send("merchant_check", state),
        Send("customer_check", state),
        Send("velocity_check", state),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel check nodes
#
# These are the same analytical logic as Phase 2.1's specialist agents,
# but now they run concurrently instead of sequentially.
# Each node is fully isolated — it receives the full state, does its own
# data fetch, and returns a partial update. No node waits for another.
# ═══════════════════════════════════════════════════════════════════════════════

async def merchant_check(state: ParallelState) -> dict:
    """
    Check 1 of 3 (parallel): Merchant fraud history.
    Simulated latency: 1.5s (fraud DB lookup).
    """
    if SIMULATE_LATENCY:
        await asyncio.sleep(SIMULATED_LATENCY["merchant_check"])

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
        "agent_findings": [{
            "agent": "merchant_check",
            "summary": summary,
            "fraud_signal": fraud_signal,
            "flags": risk_flags,
            "latency_s": SIMULATED_LATENCY["merchant_check"] if SIMULATE_LATENCY else 0,
        }],
        "evidence": [f"[MerchantCheck] {summary}"],
    }


async def customer_check(state: ParallelState) -> dict:
    """
    Check 2 of 3 (parallel): Customer spend patterns, device fingerprint,
    location history, amount anomaly. Simulated latency: 2.0s (profile service).
    """
    if SIMULATE_LATENCY:
        await asyncio.sleep(SIMULATED_LATENCY["customer_check"])

    txn = state["transaction"]
    profile = get_customer_spend_history.invoke({"customer_id": state["customer_id"]})

    risk_flags = []
    fraud_signal = 0.0

    # Device fingerprint
    known_devices = profile.get("known_devices", [])
    if txn.get("device_id") and txn["device_id"] not in known_devices:
        risk_flags.append(f"unrecognised device {txn['device_id']}")
        fraud_signal += 0.2

    # Geographic anomaly
    home_city = profile.get("home_city", "")
    recent_locs = {t.get("location", "") for t in profile.get("recent_transactions", [])}
    recent_locs.update([home_city, "Online"])
    if txn.get("location") and txn["location"] not in recent_locs:
        risk_flags.append(f"location anomaly: {txn['location']} vs home {home_city}")
        fraud_signal += 0.2

    # Amount anomaly
    avg = profile.get("avg_transaction_amount", 0)
    if avg > 0 and txn["amount"] > avg * 3:
        risk_flags.append(f"amount ${txn['amount']:.0f} is {txn['amount']/avg:.1f}× avg (${avg:.0f})")
        fraud_signal += 0.2

    # Recurring merchant (legitimate signal)
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

    summary = (
        f"Customer {state['customer_id']} ({home_city}): {'; '.join(risk_flags)}"
    )

    return {
        "customer_profile": profile,
        "agent_findings": [{
            "agent": "customer_check",
            "summary": summary,
            "fraud_signal": fraud_signal,
            "flags": risk_flags,
            "latency_s": SIMULATED_LATENCY["customer_check"] if SIMULATE_LATENCY else 0,
        }],
        "evidence": [f"[CustomerCheck] {summary}"],
    }


async def velocity_check(state: ParallelState) -> dict:
    """
    Check 3 of 3 (parallel): Card-cloning detection via rapid sequential charges.
    Key differentiator for DISP-004. Simulated latency: 1.0s (analytics query).
    """
    if SIMULATE_LATENCY:
        await asyncio.sleep(SIMULATED_LATENCY["velocity_check"])

    result = check_velocity_pattern.invoke({"customer_id": state["customer_id"]})

    fraud_signal = 0.0
    flags = []

    if result.get("is_velocity_attack"):
        fraud_signal = 0.5
        flags.append(
            f"VELOCITY ATTACK: {result['charge_count']} charges of "
            f"${result['total_amount'] / result['charge_count']:.2f} "
            f"in {result['time_span_minutes']} min (total ${result['total_amount']:.2f})"
        )
    else:
        flags.append("no velocity attack detected")

    summary = "; ".join(flags)

    return {
        "agent_findings": [{
            "agent": "velocity_check",
            "summary": summary,
            "fraud_signal": fraud_signal,
            "flags": flags,
            "latency_s": SIMULATED_LATENCY["velocity_check"] if SIMULATE_LATENCY else 0,
        }],
        "evidence": [f"[VelocityCheck] {summary}"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate node — single LLM call after all parallel checks complete
# ═══════════════════════════════════════════════════════════════════════════════

class AggregateOutput(BaseModel):
    fraud_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_factors: list[str]


AGGREGATE_SYSTEM_PROMPT = """\
You are a senior fraud analyst. You receive structured findings from 3 parallel \
specialist checks (merchant history, customer profile, velocity pattern). Each \
finding includes a fraud_signal (positive = fraud evidence, negative = legitimacy \
evidence). Combine them into a single fraud_score (0.0–1.0).

The score should reflect the weight of evidence, not just arithmetic sum. \
Cap at 1.0, floor at 0.0. A velocity attack (fraud_signal +0.5) is strong \
evidence on its own. A clean merchant (fraud_signal -0.2) combined with a \
known device and normal amount should score low.
"""


async def aggregate(state: ParallelState) -> dict:
    """
    Convergence node. Runs after ALL parallel branches have completed.
    By this point, agent_findings contains 3 dicts (one per check) merged
    via operator.add. A single LLM call produces the composite fraud score.
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

Parallel check results:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome nodes (identical to Phase 1.2 and 2.1)
# ═══════════════════════════════════════════════════════════════════════════════

def auto_credit(state: ParallelState) -> dict:
    txn = state["transaction"]
    return {
        "decision": "auto_credit",
        "resolution_amount": txn["amount"],
        "evidence": [f"AUTO-CREDIT ${txn['amount']:.2f} — score {state['fraud_score']:.2f}"],
    }


def deny(state: ParallelState) -> dict:
    return {
        "decision": "deny",
        "resolution_amount": 0.0,
        "evidence": [f"DENY — score {state['fraud_score']:.2f}, transaction appears legitimate"],
    }


def human_review(state: ParallelState) -> dict:
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


def notify(state: ParallelState) -> dict:
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

def decide_route(state: ParallelState) -> Literal["auto_credit", "deny", "human_review"]:
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
    g = StateGraph(ParallelState)

    # Nodes
    g.add_node("intake",          intake)
    g.add_node("merchant_check",  merchant_check)
    g.add_node("customer_check",  customer_check)
    g.add_node("velocity_check",  velocity_check)
    g.add_node("aggregate",       aggregate)
    g.add_node("auto_credit",     auto_credit)
    g.add_node("deny",            deny)
    g.add_node("human_review",    human_review)
    g.add_node("notify",          notify)

    # intake → fan_out dispatches all 3 checks in parallel via Send
    g.set_entry_point("intake")
    g.add_conditional_edges("intake", fan_out)

    # All 3 parallel branches converge at aggregate.
    # LangGraph buffers results until every branch has completed,
    # then calls aggregate with the merged state.
    g.add_edge("merchant_check", "aggregate")
    g.add_edge("customer_check", "aggregate")
    g.add_edge("velocity_check", "aggregate")

    # aggregate → decide → outcome → notify → END
    g.add_conditional_edges(
        "aggregate",
        decide_route,
        {"auto_credit": "auto_credit", "deny": "deny", "human_review": "human_review"},
    )
    g.add_edge("auto_credit",  "notify")
    g.add_edge("deny",         "notify")
    g.add_edge("human_review", "notify")
    g.add_edge("notify",       END)

    from langgraph.checkpoint.base import BaseCheckpointSaver
    if not isinstance(checkpointer, BaseCheckpointSaver):
        checkpointer = None
    return g.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════════════════════════════

NODE_COLORS = {
    "intake":         "blue",
    "merchant_check": "cyan",
    "customer_check": "cyan",
    "velocity_check": "magenta",
    "aggregate":      "yellow",
    "auto_credit":    "green",
    "deny":           "red",
    "human_review":   "yellow",
    "notify":         "white",
}

DECISION_COLORS = {
    "auto_credit": "green",
    "deny":        "red",
    "human_review":"yellow",
    "pending":     "dim",
}

CHECK_NODES = {"merchant_check", "customer_check", "velocity_check"}


async def run_dispute(app, dispute_id: str):
    dispute = DISPUTES[dispute_id]
    initial: ParallelState = {
        "dispute_id":       dispute_id,
        "customer_id":      dispute["customer_id"],
        "transaction":      {},
        "merchant_profile": {},
        "customer_profile": {},
        "evidence":         [],
        "agent_findings":   [],
        "fraud_score":      0.0,
        "decision":         "pending",
        "resolution_amount":0.0,
        "human_notes":      "",
        "analyst_approved": None,
        "notification_sent":False,
        "error":            None,
    }
    config = {"configurable": {"thread_id": f"2_2_{dispute_id}"}}

    console.print(f"\n[bold]Dispute {dispute_id}[/bold]  customer: {dispute['customer_id']}")
    console.print("─" * 70)

    parallel_start: float | None = None
    parallel_completed: list[str] = []

    t0 = time.perf_counter()

    async for update in app.astream(initial, config=config, stream_mode="updates"):
        for node_name, changes in update.items():
            elapsed = time.perf_counter() - t0
            color = NODE_COLORS.get(node_name, "white")

            extras = []

            if node_name == "intake":
                txn = changes.get("transaction", {})
                if txn:
                    extras.append(f"${txn['amount']:.2f} at {txn['merchant']}")
                    parallel_start = time.perf_counter()
                    console.print(
                        f"  [{color}]■[/{color}] [bold {color}]{node_name:<18}[/bold {color}]"
                        + (f"  {'  '.join(extras)}" if extras else "")
                    )
                    console.print(
                        "  [dim]  ↳ Dispatching 3 parallel checks via Send API...[/dim]"
                    )
                    continue

            elif node_name in CHECK_NODES:
                parallel_completed.append(node_name)
                check_elapsed = time.perf_counter() - (parallel_start or t0)
                finding = next(
                    (f for f in changes.get("agent_findings", []) if f["agent"] == node_name),
                    None,
                )
                signal = f"{finding['fraud_signal']:+.1f}" if finding else "?"
                extras.append(f"signal [{signal}]  [{check_elapsed:.2f}s]")

                # Show "parallel complete" banner when all 3 checks are in
                if len(parallel_completed) == 3:
                    parallel_wall = time.perf_counter() - (parallel_start or t0)
                    seq_estimate = sum(SIMULATED_LATENCY.values()) if SIMULATE_LATENCY else 0
                    console.print(
                        f"  [{color}]■[/{color}] [bold {color}]{node_name:<18}[/bold {color}]"
                        + (f"  {'  '.join(extras)}" if extras else "")
                    )
                    for item in changes.get("evidence", []):
                        console.print(f"               [dim]↳ {item}[/dim]")
                    if SIMULATE_LATENCY:
                        console.print(
                            f"\n  [bold green]✓ All 3 checks complete in {parallel_wall:.2f}s[/bold green]"
                            f"  [dim](sequential estimate: {seq_estimate:.1f}s  "
                            f"speedup: {seq_estimate/parallel_wall:.1f}×)[/dim]\n"
                        )
                    continue

            elif node_name == "aggregate" and "fraud_score" in changes:
                extras.append(f"composite score → [bold]{changes['fraud_score']:.2f}[/bold]")

            elif node_name in ("auto_credit", "deny", "human_review"):
                d = changes.get("decision", node_name)
                dc = DECISION_COLORS.get(d, "white")
                extras.append(f"[bold {dc}]{d.upper().replace('_', ' ')}[/bold {dc}]")

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
    _print_final(dispute_id, final, total_elapsed)


def _print_final(dispute_id: str, state: dict, elapsed: float):
    decision = state.get("decision", "pending")
    color = DECISION_COLORS.get(decision, "white")

    findings = state.get("agent_findings", [])

    # Findings table
    findings_table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
    findings_table.add_column("Check", style="dim", width=16)
    findings_table.add_column("Signal", justify="right", width=8)
    findings_table.add_column("Summary")
    for f in findings:
        signal_val = f.get("fraud_signal", 0)
        signal_color = "red" if signal_val > 0 else ("green" if signal_val < 0 else "dim")
        findings_table.add_row(
            f["agent"],
            f"[{signal_color}]{signal_val:+.1f}[/{signal_color}]",
            f["summary"],
        )

    # Summary table
    summary_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    summary_table.add_column("k", style="dim", width=22)
    summary_table.add_column("v")
    summary_table.add_row("Fraud score",   f"[bold]{state.get('fraud_score', 0):.2f}[/bold]")
    summary_table.add_row("Decision",      f"[bold {color}]{decision.upper().replace('_', ' ')}[/bold {color}]")
    amt = state.get("resolution_amount", 0)
    summary_table.add_row("Resolution",    f"${amt:.2f}" if amt > 0 else "—")
    summary_table.add_row("LLM calls",     "1  (aggregate only — 0 routing calls)")
    summary_table.add_row("Total elapsed", f"{elapsed:.2f}s")
    summary_table.add_row("Evidence items",str(len(state.get("evidence", []))))

    console.print()
    console.print(Panel(findings_table, title="Parallel Check Results", border_style="dim"))
    console.print(Panel(summary_table, title=f"[bold]{dispute_id} Final State[/bold]", border_style=color))


async def main():
    global SIMULATE_LATENCY

    parser = argparse.ArgumentParser()
    parser.add_argument("--dispute", choices=list(DISPUTES.keys()))
    parser.add_argument(
        "--no-simulate-latency",
        action="store_true",
        help="Skip simulated per-check latency (runs against mock data at full speed)",
    )
    args = parser.parse_args()

    if args.no_simulate_latency:
        SIMULATE_LATENCY = False

    cfg = current_config()
    latency_note = (
        f"Simulated latency: merchant={SIMULATED_LATENCY['merchant_check']}s  "
        f"customer={SIMULATED_LATENCY['customer_check']}s  "
        f"velocity={SIMULATED_LATENCY['velocity_check']}s  "
        f"(sequential estimate: {sum(SIMULATED_LATENCY.values())}s)"
        if SIMULATE_LATENCY else "No simulated latency"
    )

    console.print(Panel.fit(
        f"[bold cyan]Phase 2.2 — Parallel Fan-out (Send API)[/bold cyan]\n"
        f"intake → Send × 3 → [merchant_check ‖ customer_check ‖ velocity_check] → aggregate\n"
        f"[dim]provider: {cfg['provider']}  investigator: {cfg['investigator']}\n"
        f"{latency_note}[/dim]",
        border_style="cyan",
    ))

    async with AsyncSqliteSaver.from_conn_string("disputes.db") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        try:
            png_path = "phase2_2_parallel/graph.png"
            app.get_graph().draw_mermaid_png(output_file_path=png_path)
            console.print(f"[dim]Graph saved → {png_path}[/dim]\n")
        except Exception:
            console.print("[dim]Graph PNG skipped (paste Mermaid output into mermaid.live)[/dim]\n")

        dispute_ids = [args.dispute] if args.dispute else list(DISPUTES.keys())
        for dispute_id in dispute_ids:
            await run_dispute(app, dispute_id)

        console.print(Rule("[dim]All disputes complete[/dim]"))


if __name__ == "__main__":
    asyncio.run(main())
