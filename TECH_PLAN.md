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
| Workflow orchestration (Phase 6) | Apache Airflow 2.9+ | Batch scheduling, SLA monitoring, data refresh pipelines |
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
airflow = [
    "apache-airflow>=2.9",
    "apache-airflow-providers-postgres",  # Airflow ↔ Postgres connection
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
- **Design for Airflow (Phase 6):** Swap `SqliteSaver` → `PostgresSaver` now — Airflow also needs Postgres as its metadata DB, so a single shared Postgres instance eliminates extra infra. ChromaDB re-embedding will be triggered by the `data_refresh` DAG after daily profile updates.

#### 3.2 Human-in-the-Loop
- `interrupt()` before irreversible actions (credit issuance)
- Structured evidence brief generated for analyst
- Resume graph from `Command(resume=analyst_decision)`
- Two interrupt points: amount threshold + ambiguous score
- **Design for Airflow (Phase 6):** Disputes paused at `interrupt()` land in `decision=human_review` state. The `human_review_monitor` Airflow DAG polls Postgres hourly for disputes stuck in this state beyond SLA (4 hours) and pages the on-call analyst. Airflow handles the SLA enforcement; LangGraph handles the resumption.

#### 3.3 Streaming & Observability
- `astream_events()` for real-time terminal dashboard
- Custom callbacks for business metrics (cost/dispute, resolution rate)
- LangSmith: run trees, annotation queues, prompt hub
- **Design for Airflow (Phase 6):** Business metrics emitted here (cost/dispute, resolution rate, avg elapsed time) feed the daily `fraud_analytics_report` DAG which aggregates them and writes a summary to LangSmith datasets.

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

- **Design for Airflow (Phase 6):** The LangSmith eval dataset built here becomes the payload for the nightly `model_eval_runner` Airflow DAG. The DAG runs the eval suite, asserts accuracy ≥ 90%, and pages on-call if it regresses.

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
- [ ] Airflow `docker-compose.airflow.yml` — shared Postgres instance for both Airflow metadata and LangGraph checkpoints

---

### Phase 6 — Airflow Orchestration

> **Role split:** Airflow is the *outer* orchestrator (scheduling, retries, SLA monitoring, data pipelines). LangGraph is the *inner* reasoning engine (agent logic, state machines, checkpointing). They do not overlap.

#### 6.1 Batch Dispute Processing DAG (`dispute_batch_processor`)

- Polls Postgres every 15 minutes for disputes with `status=PENDING`
- Uses `task.expand()` (dynamic task mapping) — one Airflow task instance per dispute
- Each task invokes the LangGraph Phase 2.2 parallel graph via `asyncio.run(graph.ainvoke(...))`
- Per-dispute retry (up to 3×, exponential backoff) without retrying the full batch
- Results written back to Postgres (`status`, `decision`, `fraud_score`, `resolved_at`)

```python
# Airflow task — runs inside the DAG worker
@task(retries=3, retry_delay=timedelta(minutes=1))
def run_dispute(dispute_id: str) -> dict:
    from phase2_2_parallel.graph import build_graph
    graph = build_graph()
    result = asyncio.run(graph.ainvoke({"dispute_id": dispute_id}))
    return {"dispute_id": dispute_id, "decision": result["decision"]}

disputes = fetch_pending_disputes()
run_dispute.expand(dispute_id=disputes)   # parallel fan-out at the Airflow level
```

**Senior aspects:**
- Dynamic task mapping eliminates hand-written loops and gives per-dispute observability in the Airflow UI
- `max_active_runs=1` prevents overlapping batch runs from double-processing the same dispute
- XCom used only for small scalar results — full dispute state lives in Postgres/LangGraph checkpoint

#### 6.2 Data Refresh DAG (`data_refresh`)

- Runs nightly at 2am
- Pulls fresh merchant fraud profiles from the fraud warehouse → upserts into `merchant_profiles` table
- Pulls fresh customer spend patterns from the customer service → upserts into `customer_profiles` table
- After refresh, triggers ChromaDB re-embedding of updated profiles (Phase 3.1 link)
- Replaces static `shared/mock_data.py` in production — tools read from Postgres, not fixtures

```
refresh_merchant_profiles >> refresh_customer_profiles >> update_chromadb_embeddings
```

#### 6.3 Human Review SLA Monitor (`human_review_monitor`)

- Runs hourly
- Queries Postgres for disputes where `decision=human_review AND updated_at < now() - 4 hours`
- Sends Slack/email alert to fraud analyst on-call for each stalled dispute
- Complements Phase 3.2: LangGraph pauses at `interrupt()`, Airflow enforces the SLA and pages humans

```python
@task
def find_stalled_reviews() -> list[dict]:
    # SELECT * FROM disputes WHERE decision='human_review'
    #   AND updated_at < NOW() - INTERVAL '4 hours'
    ...

@task
def send_alerts(stalled: list[dict]):
    for dispute in stalled:
        notify_slack(f"[SLA BREACH] {dispute['id']} has been in human_review for > 4h")
```

#### 6.4 Model Evaluation DAG (`model_eval_runner`)

- Runs nightly at 3am (after data_refresh completes)
- Executes the LangSmith eval dataset built in Phase 4 against the current production graph
- Asserts accuracy ≥ 90% — fails the DAG (triggers on-call alert) if score regresses
- Produces a daily accuracy trend report stored in LangSmith

#### 6.5 Fraud Analytics Report DAG (`fraud_analytics_report`)

- Runs daily at 6am
- Aggregates prior day metrics: disputes processed, auto-credit rate, deny rate, human-review rate, avg resolution time, avg cost per dispute
- Writes summary to LangSmith annotation queue for product review

**Senior aspects across Phase 6:**
- Airflow DAGs are tested independently of LangGraph graph logic (unit test DAG structure; integration test the full pipeline)
- `docker-compose.airflow.yml` mounts the project directory so Airflow workers can import LangGraph graphs directly — no separate package needed
- Shared Postgres: one instance serves both Airflow metadata DB and LangGraph `PostgresSaver` checkpoint store
- Airflow Connections store DB credentials and Slack webhook — not duplicated in `.env`

---

### Phase 6 — Project Structure Addition

```
airflow/
├── dags/
│   ├── dispute_batch_processor.py   ← Phase 6.1
│   ├── data_refresh.py              ← Phase 6.2
│   ├── human_review_monitor.py      ← Phase 6.3
│   ├── model_eval_runner.py         ← Phase 6.4
│   └── fraud_analytics_report.py   ← Phase 6.5
└── plugins/
    └── operators/
        └── langgraph_operator.py    ← custom BaseOperator wrapping graph.ainvoke()
docker-compose.airflow.yml           ← Airflow webserver + scheduler + shared Postgres
```

---

## Project Structure

```
fraud-disputes-resolution/
├── TECH_PLAN.md                     ← this file
├── FRAUD_DISPUTE_REQUIREMENTS.md    ← business requirements
├── pyproject.toml
├── .env.example
├── docker-compose.airflow.yml       ← Phase 6: Airflow + shared Postgres
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
├── airflow/                         ← Phase 6: Airflow DAGs + operators
│   ├── dags/
│   │   ├── dispute_batch_processor.py
│   │   ├── data_refresh.py
│   │   ├── human_review_monitor.py
│   │   ├── model_eval_runner.py
│   │   └── fraud_analytics_report.py
│   └── plugins/
│       └── operators/
│           └── langgraph_operator.py
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
