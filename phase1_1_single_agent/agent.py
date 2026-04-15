"""
Phase 1.1 — Single Agent with Tools (3 implementation patterns)

TOKEN COST COMPARISON
─────────────────────────────────────────────────────────────────
Pattern 1 — ReAct (sequential):
  The LLM decides which tool to call one at a time. Every turn re-sends
  the entire message history (system + all prior tool calls + results).
  Cost grows quadratically with number of tool calls.

  Turn 1: system + user                       ~600 tokens
  Turn 2: system + user + tool1 + result1    ~1,200 tokens
  Turn 3: +tool2 + result2                   ~2,000 tokens
  Turn 4: +tool3 + result3                   ~2,800 tokens
  Turn 5: final answer                        ~3,200 tokens
  Per dispute: ~10–15k tokens × 4 = ~50k total

Pattern 2 — Parallel tool calling:
  LLM calls ALL tools in a single turn. One tool-execution round,
  then one reasoning pass. History is shorter.
  Per dispute: ~4–6k tokens × 4 = ~20k total

Pattern 3 — Gather-then-Reason (recommended for known schemas):
  Tools are called deterministically (no LLM). One single LLM call
  with ALL evidence pre-loaded. Zero redundant context.
  Per dispute: ~1.5–2.5k tokens × 4 = ~8k total

Run:
    uv run python -m phase1_1_single_agent.agent               # all 3 patterns
    uv run python -m phase1_1_single_agent.agent --mode react
    uv run python -m phase1_1_single_agent.agent --mode parallel
    uv run python -m phase1_1_single_agent.agent --mode gather
"""

import argparse
import asyncio
import json
import logging
import os
import warnings
from dataclasses import dataclass, field

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

warnings.filterwarnings("ignore", message="create_react_agent has been moved", category=DeprecationWarning)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.prebuilt import create_react_agent

from shared.mock_data import DISPUTES, TRANSACTIONS, MERCHANT_PROFILES, CUSTOMER_PROFILES, VELOCITY_CHARGES
from shared.model_factory import get_investigator, current_config
from shared.models import FraudDecision
from shared.tools import (
    get_transaction, check_merchant_fraud_history,
    get_customer_spend_history, check_velocity_pattern,
    ALL_TOOLS,
)

console = Console()

# ── LangSmith guard ──────────────────────────────────────────────────────────
_ls_key = os.getenv("LANGCHAIN_API_KEY", "")
if _ls_key and not _ls_key.startswith(("ls__", "lsv2_")):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    console.print("[dim yellow]⚠  LangSmith key format unrecognised — tracing disabled.[/dim yellow]")


# ── Token tracking ───────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Accumulates token counts across all LLM calls in one investigation."""
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    call_log: list[dict] = field(default_factory=list)  # per-call breakdown

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, label: str, input_t: int, output_t: int):
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.llm_calls += 1
        self.call_log.append({"label": label, "input": input_t, "output": output_t})


class TokenTracker(BaseCallbackHandler):
    """
    Provider-agnostic token counter using LangChain's callback system.
    Works with OpenAI and Anthropic — both populate response_metadata.
    """
    def __init__(self, usage: TokenUsage):
        self.usage = usage

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                meta = getattr(gen.message, "response_metadata", {}) or {}
                # OpenAI format
                if "token_usage" in meta:
                    tu = meta["token_usage"]
                    self.usage.record(
                        "llm_call",
                        tu.get("prompt_tokens", 0),
                        tu.get("completion_tokens", 0),
                    )
                # Anthropic format
                elif "usage" in meta:
                    u = meta["usage"]
                    self.usage.record(
                        "llm_call",
                        u.get("input_tokens", 0),
                        u.get("output_tokens", 0),
                    )

    def on_tool_end(self, output, **kwargs):
        self.usage.tool_calls += 1


# ── Prompts ───────────────────────────────────────────────────────────────────

# Used by ReAct and Parallel — LLM needs instructions on what tools to call
REACT_SYSTEM_PROMPT = """\
You are a fraud investigator for a credit card company.

Scoring signals (additive):
  +0.3  Merchant > 20 prior disputes
  +0.2  Merchant fraud rate > 15%
  +0.2  Transaction location ≠ customer's known locations
  +0.2  Unrecognised device
  +0.2  Amount > 3× customer average
  +0.4  Known card skimmer at merchant
  +0.5  Velocity attack (3+ charges in 2h at same merchant)
  -0.4  Active subscription/recurring charge at same merchant
  -0.2  Merchant has 0 disputes and < 1% fraud rate

Decision thresholds:
  score > 0.8 AND amount < 500  → auto_credit
  score > 0.8 AND amount ≥ 500  → human_review
  0.4 ≤ score ≤ 0.8             → human_review
  score < 0.4                   → deny
"""

# Used by Gather-then-Reason — no tool instructions needed, just reasoning
REASON_SYSTEM_PROMPT = """\
You are a fraud investigator. All evidence has been pre-fetched and is in the user message.
Apply the scoring signals and thresholds below to produce a FraudDecision.

Scoring signals (additive):
  +0.3  Merchant > 20 prior disputes
  +0.2  Merchant fraud rate > 15%
  +0.2  Transaction location ≠ customer's known locations
  +0.2  Unrecognised device
  +0.2  Amount > 3× customer average
  +0.4  Known card skimmer at merchant
  +0.5  Velocity attack (3+ charges in 2h at same merchant)
  -0.4  Active subscription/recurring charge at same merchant
  -0.2  Merchant has 0 disputes and < 1% fraud rate

Decision thresholds:
  score > 0.8 AND amount < 500  → auto_credit
  score > 0.8 AND amount ≥ 500  → human_review
  0.4 ≤ score ≤ 0.8             → human_review
  score < 0.4                   → deny
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern 1 — ReAct (sequential tool calls)
# ═══════════════════════════════════════════════════════════════════════════════

async def investigate_react(dispute_id: str, usage: TokenUsage) -> FraudDecision:
    """
    Classic ReAct loop: LLM decides each tool call one at a time.
    History grows with every turn — most expensive pattern.
    Best used when the tool sequence is unknown/dynamic upfront.
    """
    dispute = DISPUTES[dispute_id]
    tracker = TokenTracker(usage)
    llm = get_investigator(max_tokens=2048, callbacks=[tracker])

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=REACT_SYSTEM_PROMPT,
        response_format=FraudDecision,
    )

    result = await agent.ainvoke({
        "messages": [HumanMessage(
            content=(
                f"Investigate dispute {dispute['dispute_id']}. "
                f"Transaction: {dispute['transaction_id']}. "
                f"Customer: {dispute['customer_id']}."
            )
        )]
    })
    return result["structured_response"]


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern 2 — Parallel tool calling
# ═══════════════════════════════════════════════════════════════════════════════

async def investigate_parallel(dispute_id: str, usage: TokenUsage) -> FraudDecision:
    """
    Ask the LLM to call ALL tools in a single turn using parallel tool calling.
    Reduces message history to: system + user → [tool calls] → results → decision.
    Best when you want LLM control over which tools to call but want fewer turns.
    """
    dispute = DISPUTES[dispute_id]
    tracker = TokenTracker(usage)
    llm = get_investigator(max_tokens=2048, callbacks=[tracker])
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    llm_with_output = llm.with_structured_output(FraudDecision)

    # Turn 1: LLM decides all tool calls at once
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Investigate dispute {dispute['dispute_id']}. "
            f"Transaction: {dispute['transaction_id']}. "
            f"Customer: {dispute['customer_id']}. "
            f"Call all relevant tools now."
        )},
    ]

    ai_msg: AIMessage = await llm_with_tools.ainvoke(messages)
    messages.append(ai_msg)

    # Execute every tool call the LLM requested
    for tool_call in ai_msg.tool_calls:
        tool_fn = next(t for t in ALL_TOOLS if t.name == tool_call["name"])
        result = tool_fn.invoke(tool_call["args"])
        usage.tool_calls += 1
        messages.append(ToolMessage(
            content=json.dumps(result),
            tool_call_id=tool_call["id"],
        ))

    # Turn 2: one reasoning pass over all results → structured output
    messages.append({"role": "user", "content": "Now produce your FraudDecision."})
    decision: FraudDecision = await llm_with_output.ainvoke(messages)
    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern 3 — Gather-then-Reason
# ═══════════════════════════════════════════════════════════════════════════════

def _gather_evidence(dispute_id: str) -> tuple[dict, str]:
    """
    Fetch all data deterministically — zero LLM tokens.
    Returns (raw data dict, formatted evidence string for the prompt).
    """
    dispute = DISPUTES[dispute_id]
    txn       = get_transaction.invoke({"transaction_id": dispute["transaction_id"]})
    merchant  = check_merchant_fraud_history.invoke({"merchant_id": txn["merchant_id"]})
    customer  = get_customer_spend_history.invoke({"customer_id": dispute["customer_id"]})
    velocity  = check_velocity_pattern.invoke({"customer_id": dispute["customer_id"]})

    evidence_text = f"""
DISPUTE ID: {dispute_id}

TRANSACTION:
{json.dumps(txn, indent=2)}

MERCHANT RISK PROFILE:
{json.dumps(merchant, indent=2)}

CUSTOMER PROFILE:
{json.dumps(customer, indent=2)}

VELOCITY CHECK:
{json.dumps(velocity, indent=2)}
""".strip()

    return {"txn": txn, "merchant": merchant, "customer": customer, "velocity": velocity}, evidence_text


async def investigate_gather(dispute_id: str, usage: TokenUsage) -> FraudDecision:
    """
    Gather-then-Reason: fetch all data with zero LLM calls, then make
    exactly ONE LLM call to reason over the evidence.
    Best when the data schema is known upfront — cheapest pattern.
    """
    tracker = TokenTracker(usage)
    llm = get_investigator(max_tokens=1024, callbacks=[tracker])
    llm_with_output = llm.with_structured_output(FraudDecision)

    # Step 1: gather — no LLM, no tokens
    _, evidence_text = _gather_evidence(dispute_id)
    usage.tool_calls += 4  # 4 deterministic fetches

    # Step 2: reason — exactly ONE LLM call
    decision: FraudDecision = await llm_with_output.ainvoke([
        {"role": "system", "content": REASON_SYSTEM_PROMPT},
        {"role": "user",   "content": evidence_text},
    ])
    return decision


# ═══════════════════════════════════════════════════════════════════════════════
# CLI runner
# ═══════════════════════════════════════════════════════════════════════════════

PATTERNS = {
    "react":    ("Pattern 1 — ReAct (sequential)",    investigate_react),
    "parallel": ("Pattern 2 — Parallel tool calling", investigate_parallel),
    "gather":   ("Pattern 3 — Gather-then-Reason",    investigate_gather),
}

DISPUTE_IDS = ["DISP-001", "DISP-002", "DISP-003", "DISP-004"]


async def run_pattern(name: str, label: str, fn):
    cfg = current_config()
    console.rule(f"[bold cyan]{label}[/bold cyan]")
    console.print(f"[dim]provider: {cfg['provider']}  model: {cfg['investigator']}[/dim]\n")

    total = TokenUsage()

    for dispute_id in DISPUTE_IDS:
        usage = TokenUsage()
        console.print(f"[bold yellow]▶ {dispute_id}[/bold yellow]", end="  ")
        try:
            decision = await fn(dispute_id, usage)
            total.input_tokens  += usage.input_tokens
            total.output_tokens += usage.output_tokens
            total.llm_calls     += usage.llm_calls
            total.tool_calls    += usage.tool_calls
            _print_decision(decision, usage)
        except Exception as e:
            console.print(f"[red]✗ failed: {e}[/red]")

    _print_summary(label, total)
    return total


def _print_decision(decision: FraudDecision, usage: TokenUsage):
    colors = {"auto_credit": "green", "deny": "red", "human_review": "yellow"}
    color = colors.get(decision.decision, "white")
    console.print(
        f"score=[bold]{decision.fraud_score:.2f}[/bold]  "
        f"decision=[bold {color}]{decision.decision.upper().replace('_',' ')}[/bold {color}]  "
        f"[dim]tokens: {usage.input_tokens}in + {usage.output_tokens}out = {usage.total_tokens} "
        f"({usage.llm_calls} LLM call{'s' if usage.llm_calls != 1 else ''})[/dim]"
    )


def _print_summary(label: str, total: TokenUsage):
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total input tokens",  str(total.input_tokens))
    table.add_row("Total output tokens", str(total.output_tokens))
    table.add_row("Total tokens",        f"[bold]{total.total_tokens}[/bold]")
    table.add_row("LLM calls",           str(total.llm_calls))
    table.add_row("Tool calls",          str(total.tool_calls))
    console.print(Panel(table, title=f"[bold]{label} — 4-dispute totals[/bold]", border_style="cyan"))
    console.print()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["react", "parallel", "gather", "all"],
        default="all",
        help="Which investigation pattern to run (default: all — runs all 3 for comparison)",
    )
    args = parser.parse_args()

    modes = list(PATTERNS.keys()) if args.mode == "all" else [args.mode]
    totals: dict[str, TokenUsage] = {}

    for mode in modes:
        label, fn = PATTERNS[mode]
        totals[mode] = await run_pattern(mode, label, fn)

    if len(modes) > 1:
        _print_comparison(totals)


def _print_comparison(totals: dict[str, TokenUsage]):
    console.rule("[bold]Token Cost Comparison — All Patterns[/bold]")
    table = Table(header_style="bold", padding=(0, 2))
    table.add_column("Pattern")
    table.add_column("Total tokens", justify="right")
    table.add_column("LLM calls", justify="right")
    table.add_column("vs. ReAct", justify="right")

    react_total = totals["react"].total_tokens if "react" in totals else 1
    for mode, (label, _) in PATTERNS.items():
        if mode not in totals:
            continue
        t = totals[mode]
        ratio = t.total_tokens / react_total if react_total else 1
        vs = "baseline" if mode == "react" else f"{ratio:.1%} of ReAct"
        table.add_row(label, str(t.total_tokens), str(t.llm_calls), vs)

    console.print(table)
    console.print(
        "\n[dim]Gather-then-Reason wins for fixed schemas. "
        "Use ReAct only when tool sequence is dynamic.[/dim]"
    )


if __name__ == "__main__":
    asyncio.run(main())
