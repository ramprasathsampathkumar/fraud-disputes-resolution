# Technical Plan — LangGraph / LangChain / LangSmith

> Learning path: junior → senior engineer patterns, local-first, production-ready trajectory.
> Domain: Credit card fraud dispute resolution (see FRAUD_DISPUTE_REQUIREMENTS.md for business context).

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| LLM | Claude via `langchain-anthropic` | `claude-sonnet-4-6` for reasoning, `claude-haiku-4-5` for fast classification |
| Orchestration | `langgraph >= 0.2` | StateGraph, Send API, subgraphs, interrupt |
| Chains / tools | `langchain >= 0.3` | LCEL, ToolNode, structured output |
| Short-term memory | `langgraph-checkpoint-sqlite` | local dev; swap to postgres in prod |
| Long-term memory | `chromadb` | merchant + customer pattern accumulation |
| Observability | `langsmith` | tracing, eval, prompt versioning |
| Validation | `pydantic v2` | models at every system boundary |
| Runtime | Python 3.11+, `uv` | fast dependency resolution |
| API layer (Phase 5+) | `fastapi` + `uvicorn` | REST + SSE streaming endpoints |
| Containerisation | Docker + docker-compose | local integration, staging parity |

---

## Dependency Versions (pyproject.toml targets)

```toml
[project]
name = "fraud-disputes-resolution"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langgraph-checkpoint-sqlite",
    "langchain>=0.3",
    "langchain-anthropic",
    "langchain-core",
    "langsmith",
    "chromadb>=0.5",
    "pydantic>=2.0",
    "python-dotenv",
    "rich",          # terminal output / streaming display
]

[project.optional-dependencies]
prod = [
    "langgraph-checkpoint-postgres",
    "asyncpg",
    "fastapi",
    "uvicorn[standard]",
]
dev = [
    "pytest",
    "pytest-asyncio",
    "pytest-langsmith",  # LangSmith eval harness
]
```

---

## LangSmith Setup (Free Tier)

1. Create account at smith.langchain.com — free tier: 5k traces/month, 14-day retention.
2. Set env vars:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=<your key>
   LANGCHAIN_PROJECT=fraud-disputes-resolution
   ```
3. Every LangChain/LangGraph run auto-traces — no code changes needed.
4. Progression: traces → datasets → evaluators → CI regression gates.

---

## Phase Build Sequence

### Phase 1 — LangChain Fundamentals

#### 1.1 Single Agent with Tools
- `DisputeInvestigatorAgent` using ReAct pattern
- 3 tools: `get_transaction`, `check_merchant_fraud_history`, `get_customer_spend_history`
- Structured output via Pydantic (`FraudDecision`)
- LangSmith tracing enabled from day one

**Senior aspects:**
- `with_structured_output()` + validation error recovery
- `RunnableRetry` for flaky tool calls
- Async tool execution (`ainvoke`)

#### 1.2 LangGraph State Machine
- Rebuild 1.1 as `StateGraph` with typed `DisputeState`
- Nodes: `intake → enrich → score → decide → notify`
- Conditional edges off `decide` node
- `SqliteSaver` checkpointer — every state transition persisted

**Senior aspects:**
- State schema design (additive lists via `Annotated[list, operator.add]`)
- Graph visualization (`draw_mermaid_png`)
- Deterministic vs. LLM-driven routing tradeoffs

---

### Phase 2 — Multi-Agent Patterns

#### 2.1 Supervisor Pattern
- `DisputeSupervisor` orchestrates specialist subagents
- Agents: `MerchantIntelAgent`, `CustomerProfileAgent`, `VelocityAgent`
- Supervisor uses structured output to route, not free-text parsing

**Senior aspects:**
- Avoiding supervisor bottlenecks (don't funnel all data through it)
- Subagent output schemas — contract between orchestrator and worker
- Fallback edges when specialist returns empty/error

#### 2.2 Parallel Fan-out (Send API)
- Fan out 4 checks simultaneously using `Send`
- Merge partial scores into composite fraud score
- Demonstrate wall-clock time: ~2s parallel vs ~8s sequential

**Senior aspects:**
- `operator.add` reducer for parallel result accumulation
- Handling partial failures — one branch fails, others proceed
- Map-reduce pattern for variable-length inputs

---

### Phase 3 — Production Patterns

#### 3.1 Persistence & Long-term Memory
- `SqliteSaver` for resumable dispute threads
- ChromaDB for cross-dispute learning (merchant profiles, customer travel patterns)
- Show memory changing outcomes over time

#### 3.2 Human-in-the-Loop
- `interrupt()` before irreversible actions (credit issuance)
- Structured evidence brief generated for analyst
- Resume graph from `Command(resume=analyst_decision)`
- Two interrupt points: amount threshold + ambiguous score

#### 3.3 Streaming & Observability
- `astream_events()` for real-time terminal dashboard
- Custom callbacks for business metrics (cost/dispute, resolution rate)
- LangSmith: run trees, annotation queues, prompt hub

#### 3.4 Error Handling & Retries
- Pydantic validation at all system boundaries
- `RunnableRetry` with exponential backoff on tool calls
- Error → human_review fallback edge (never auto-decide on incomplete evidence)
- LLM output validation — re-prompt if fraud_score outside `[0.0, 1.0]`

---

### Phase 4 — Testing

```
Unit:         Test each graph node in isolation with mock DisputeState
Integration:  Run full graph paths against all 4 demo disputes
LangSmith:    Eval datasets with ground-truth decisions, LLM-as-judge scoring
CI:           Regression gate — new prompt/model must match baseline on eval dataset
```

---

### Phase 5 — Deployment

#### Target: LangGraph Platform (Cloud)
- Managed checkpointing (no infra for Postgres)
- Built-in streaming API + background run queues
- Auth + multi-tenancy
- Native LangSmith integration
- Free dev tier; scales on usage

#### Fallback: Self-hosted
```
Railway / Render:   Docker container + managed Postgres (simple, cheap)
AWS ECS + RDS:      Enterprise path — ECS Fargate, RDS Postgres, ALB
```

#### Production checklist
- [ ] Swap `SqliteSaver` → `PostgresSaver`
- [ ] Secrets via env vars / AWS Secrets Manager
- [ ] FastAPI wrapper with `/disputes` POST + SSE streaming endpoint
- [ ] Docker multi-stage build (builder + runtime layers)
- [ ] LangSmith project per environment (dev / staging / prod)
- [ ] Eval regression CI step on every PR

---

## Project Structure

```
fraud-disputes-resolution/
├── TECH_PLAN.md                     ← this file
├── FRAUD_DISPUTE_REQUIREMENTS.md    ← business requirements
├── pyproject.toml
├── .env.example
├── data/
│   └── disputes.json                ← 4 mock dispute scenarios
├── shared/
│   ├── models.py                    ← Pydantic: Transaction, DisputeState, FraudDecision
│   ├── mock_data.py                 ← dispute + customer + merchant fixtures
│   └── tools.py                     ← shared tool definitions
├── phase1_1_single_agent/
│   └── agent.py                     ← ReAct agent, structured output
├── phase1_2_state_machine/
│   └── graph.py                     ← StateGraph, conditional edges, checkpointer
├── phase2_1_supervisor/
│   └── graph.py
├── phase2_2_parallel/
│   └── graph.py
├── phase3_1_persistence/
│   └── graph.py
├── phase3_2_human_in_loop/
│   └── graph.py
├── phase3_3_streaming/
│   └── graph.py
├── phase3_4_error_handling/
│   └── graph.py
└── tests/
    ├── unit/
    ├── integration/
    └── evals/                       ← LangSmith eval datasets + runners
```

---

## Key LangGraph Concepts (Cheat Sheet)

```python
# State — always TypedDict, use Annotated reducers for lists
class DisputeState(TypedDict):
    evidence: Annotated[list[str], operator.add]  # parallel nodes append safely

# Graph construction
graph = StateGraph(DisputeState)
graph.add_node("intake", intake_node)
graph.add_conditional_edges("decide", routing_fn, {"auto_credit": ..., "deny": ...})
graph.set_entry_point("intake")

# Checkpointing
checkpointer = SqliteSaver.from_conn_string("disputes.db")
app = graph.compile(checkpointer=checkpointer)

# Invoke with thread_id for persistence
config = {"configurable": {"thread_id": "DISP-001"}}
result = await app.ainvoke(initial_state, config=config)

# Human-in-the-loop
app = graph.compile(checkpointer=checkpointer, interrupt_before=["issue_credit"])
await app.ainvoke(state, config)          # pauses at issue_credit
await app.ainvoke(Command(resume={"approved": True}), config)  # resumes

# Parallel fan-out
def fan_out(state) -> list[Send]:
    return [Send("check_merchant", state), Send("check_velocity", state)]
graph.add_conditional_edges("router", fan_out)
```
