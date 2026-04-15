# Phase 1.1 Findings — Token Optimization Patterns

## The Problem: ReAct's Cumulative Context Cost

The standard ReAct (Reason + Act) loop re-sends the **entire message history** on every LLM turn. Cost grows with each tool call:

```
Turn 1: system + user                                    ~600 tokens
Turn 2: system + user + tool1_call + tool1_result      ~1,200 tokens
Turn 3: + tool2_call + tool2_result                    ~2,000 tokens
Turn 4: + tool3_call + tool3_result                    ~2,800 tokens
Turn 5: final answer                                   ~3,200 tokens
                                              ─────────────────────
                               per dispute:  ~10–15k tokens
                               × 4 disputes: ~50k tokens total
```

---

## Benchmark Results (gpt-4o-mini, 4 disputes)

| Pattern | Total Tokens | LLM Calls | Cost vs ReAct |
|---|---|---|---|
| Pattern 1 — ReAct (sequential) | 24,334 | 18 | baseline |
| Pattern 2 — Parallel tool calling | 7,719 | 8 | **−68%** |
| Pattern 3 — Gather-then-Reason | 4,705 | 4 | **−81%** |

---

## Pattern Breakdown

### Pattern 1 — ReAct (Sequential)
**How it works:** LLM decides which tool to call one at a time. Each turn appends to message history and resends the full context.

```
LLM → call tool_1 → result appended → LLM again → call tool_2 → ...
```

**Cost driver:** System prompt (~500 tokens) resent on every turn. History grows quadratically.

**When to use:** Tool sequence is unknown upfront — open-ended research agents, agents that branch based on intermediate findings.

---

### Pattern 2 — Parallel Tool Calling
**How it works:** LLM calls all tools in a single turn. One tool-execution round, then one reasoning pass.

```
LLM → [call tool_1, tool_2, tool_3, tool_4 simultaneously] → all results → LLM reasons once
```

**Cost driver:** Two LLM calls per dispute (tool selection + reasoning) instead of 4–5. History stays flat.

**When to use:** Tool sequence is known, but you want the LLM to decide which subset of tools to call (optional tools, conditional fetches).

---

### Pattern 3 — Gather-then-Reason ✓ Recommended
**How it works:** All tools are called deterministically (no LLM involved). One single LLM call receives all pre-fetched evidence and produces the structured output.

```
[fetch transaction]        ← no LLM, no tokens
[fetch merchant history]   ← no LLM, no tokens
[fetch customer profile]   ← no LLM, no tokens
[fetch velocity data]      ← no LLM, no tokens
         ↓
LLM receives all evidence → FraudDecision (1 call)
```

**Cost driver:** Exactly 1 LLM call per dispute regardless of how many data sources are consulted.

**When to use:** Data schema is fixed and always the same — fraud investigation, document processing, structured data pipelines. The tool sequence does not change based on intermediate findings.

---

## Decision Guide

```
Is the tool sequence dynamic (depends on prior results)?
  YES → ReAct
  NO  →
        Do you need the LLM to decide which tools to skip?
          YES → Parallel
          NO  → Gather-then-Reason  ← use this by default
```

---

## Implementation Notes

### Token tracking (provider-agnostic)
Used a `BaseCallbackHandler` that reads `response_metadata` from `AIMessage`:
- OpenAI: `response_metadata["token_usage"]["prompt_tokens"]`
- Anthropic: `response_metadata["usage"]["input_tokens"]`

This works without any provider-specific SDK — the LangChain abstraction normalises it.

### Prompt sizing matters
The ReAct system prompt (~500 tokens) is resent on every LLM turn. In Gather-then-Reason it's sent once. For a 5-turn ReAct loop that's 2,500 tokens just for the system prompt vs 500. Keep prompts concise.

### Output token budget
Setting `max_tokens=1024` on the Gather-then-Reason LLM (vs 4096 on ReAct) is safe — the single reasoning call doesn't need a large generation budget. Tighten this per pattern.

---

## What We Carry Forward to Phase 1.2

- **Gather-then-Reason is the default pattern** for all StateGraph nodes going forward
- Token tracking via `BaseCallbackHandler` will be wired into the graph's shared state
- Each graph node will log its token cost independently — enabling per-node cost attribution in Phase 3.3 (observability)
