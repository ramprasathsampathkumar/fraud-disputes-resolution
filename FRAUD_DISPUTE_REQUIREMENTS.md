# Credit Card Fraud Dispute Resolution — Agentic System

> A production-grade multi-agent system built with LangGraph + Claude (Anthropic) that automates credit card fraud dispute investigation and resolution. Designed to demonstrate the full value of agentic AI systems across progressively complex patterns.

---

## The Problem

When a credit card customer disputes a charge today:

1. They call the bank or tap "Dispute" in the app
2. A human analyst manually looks up the transaction, checks merchant history, reviews the customer's spend profile
3. The customer waits **7–10 business days** for a resolution letter

A large bank processes **thousands of disputes per day**. The majority are straightforward — a merchant the customer has never visited, in a city they've never been to, flagged by dozens of prior customers. These should resolve in **seconds, not days**.

This system automates the investigation and decision pipeline while keeping humans in the loop for high-value or ambiguous cases.

---

## Demo Scenarios

Four disputes, each designed to exercise a different code path:

| ID | Amount | Merchant | Expected Path |
|---|---|---|---|
| `DISP-001` | $340.00 | Biscayne Grill, Miami FL | **Auto-credit** — merchant has 47 prior fraud disputes, customer has never been to Florida |
| `DISP-002` | $1,200.00 | Apple Store Online | **Human review** — customer buys Apple products regularly but amount is unusually high (> $500 threshold) |
| `DISP-003` | $18.50 | Netflix | **Deny** — customer has an active Netflix subscription, charge is legitimate |
| `DISP-004` | $340.00 × 5 | Shell Gas Station (5 charges in 2 hours) | **Fraud ring / velocity attack** — card cloned at a skimmer, multiple rapid charges |

---

## System Architecture

```
Customer Disputes Transaction
            ↓
     Intake & Enrichment
            ↓
   ┌────────┼────────┐
   ↓        ↓        ↓        (parallel — Phase 2.2)
Merchant  Customer  Velocity &
Intel     Profile   Location
Analysis  Analysis  Analysis
   └────────┼────────┘
            ↓
     Fraud Score & Decision
            ↓
   ┌────────┴────────┐
   ↓                 ↓
Auto-Credit      Human Review
(score > 0.8,    (score 0.4–0.8,
 amount < $500)   or amount ≥ $500)
   ↓                 ↓
Customer         Analyst reviews evidence
Notified         brief → approves/denies
                         ↓
                  Customer Notified
```

---

## Fraud Score Thresholds

| Score Range | Meaning | Action |
|---|---|---|
| `> 0.8` | High confidence fraud | Auto-credit (if amount < $500) |
| `0.4 – 0.8` | Ambiguous | Route to human review |
| `< 0.4` | Likely legitimate | Deny dispute |
| Any score | Amount ≥ $500 | Always route to human review |

---

## Shared State (LangGraph)

```python
class DisputeState(TypedDict):
    dispute_id: str
    transaction: dict           # amount, merchant, merchant_id, date, location
    customer_profile: dict      # customer_id, home_city, recent_transactions
    evidence: list[str]         # accumulated findings from each agent/check
    fraud_score: float          # 0.0 → 1.0, built up across checks
    decision: str               # "auto_credit" | "deny" | "human_review" | "pending"
    resolution_amount: float    # amount to credit back
    human_notes: str            # analyst input when human-in-the-loop fires
    notification_sent: bool
```

---

## Phase-by-Phase Build Plan

### Phase 1.1 — Single Agent with Tools

**What it does:** One `DisputeInvestigatorAgent` investigates a dispute end-to-end using 3 tools.

**Tools:**

```python
@tool
def get_transaction(transaction_id: str) -> dict:
    """Fetch full transaction details: merchant, amount, location, timestamp."""

@tool
def check_merchant_fraud_history(merchant_id: str) -> dict:
    """Returns prior dispute count, fraud rate %, and known scam flags for a merchant."""

@tool
def get_customer_spend_history(customer_id: str, days: int = 90) -> list[dict]:
    """Returns recent transactions to establish normal spending geography and patterns."""
```

**Expected output for DISP-001:**
```
Merchant 'Biscayne Grill Miami' has 47 prior disputes (fraud rate: 34%).
Customer CUST-4821 has zero transactions in Florida in the past 90 days.
Recommendation: AUTO-CREDIT $340.00
```

---

### Phase 1.2 — LangGraph State Machine

**What it does:** Rebuilds the Phase 1.1 agent as a typed state graph with explicit lifecycle stages.

**Graph nodes:**

| Node | Responsibility |
|---|---|
| `intake` | Parse dispute, load transaction details |
| `enrich` | Add customer profile and merchant data to state |
| `score` | Compute fraud score from accumulated evidence |
| `decide` | Branch: auto_credit / deny / human_review |
| `notify` | Send resolution to customer |

**Edges:**
- `score → decide` is always sequential
- `decide → auto_credit` if `fraud_score > 0.8 AND amount < 500`
- `decide → human_review` if `fraud_score in [0.4, 0.8] OR amount >= 500`
- `decide → deny` if `fraud_score < 0.4`

**Demo value:** Watch the `evidence` list and `fraud_score` field update at each node as the state flows through the graph.

---

### Phase 2.1 — Supervisor Pattern

**What it does:** A `DisputeSupervisor` receives the dispute and routes to the right specialist agent based on the transaction characteristics.

**Specialist agents:**

| Agent | Triggered when | Responsibility |
|---|---|---|
| `MerchantIntelligenceAgent` | Always runs first | Prior disputes, fraud rate, known skimmer flags |
| `CustomerProfileAgent` | Always runs second | Travel history, device fingerprint, normal spend geography |
| `VelocityAgent` | Multiple charges detected | Card-cloning patterns, rapid sequential charges |

**Demo value:** Feed all 4 disputes. `DISP-004` (Shell Gas × 5) gets routed to `VelocityAgent` which identifies the cloning pattern. Others follow the standard two-agent path.

---

### Phase 2.2 — Parallel Subgraph Execution

**What it does:** For any dispute, fan out 4 checks simultaneously using LangGraph's `Send` API, then merge results into a composite fraud score.

**Parallel checks:**

```
            Dispute Received
                  ↓
             [Send API]
   ┌──────┬──────┬──────┬──────┐
   ↓      ↓      ↓      ↓      ↓
 Merch  Custmr  Veloc  Loctn  (future: dark-web check)
 Check  Pattern Check  Check
   └──────┴──────┴──────┴──────┘
                  ↓
           Merge → Composite Fraud Score
```

**Demo value:**
- Show elapsed time: 4 checks in ~2s parallel vs. ~8s sequential
- `DISP-002` (Apple $1,200): all 4 checks complete, results conflict (customer is a regular buyer, but amount and location are unusual) → ambiguous score → human review triggered

---

### Phase 3.1 — Persistence & Memory

**What it does:** Disputes persist across sessions. Long-term memory accumulates patterns over time.

**Short-term (SqliteSaver):**
- Every dispute is a resumable graph thread keyed by `dispute_id`
- An analyst can close their laptop mid-investigation and resume the next morning with full state intact

**Long-term (ChromaDB):**
- After resolving disputes, the system stores learnings:
  - *"Merchant MCH-0042 (Biscayne Grill Miami) — 47 confirmed fraud cases. Fast-track all future disputes."*
  - *"CUST-4821 travels to Miami every April for a conference."* ← prevents false positive next year
- Merchant risk profiles persist and improve over time

**Demo value:**
- Run `DISP-001` cold → fraud_score: `0.91` → auto-credit
- Seed memory with *"CUST-4821 travels to Miami in April"*
- Re-run `DISP-001` → fraud_score drops to `0.55` → human review triggered
- Shows memory meaningfully changes outcomes

---

### Phase 3.2 — Human-in-the-Loop

**What it does:** Pauses graph execution at two points for human input before taking irreversible actions.

**Interrupt point 1 — Amount threshold:**
```
interrupt_before: "issue_credit"
Condition: resolution_amount >= 500
```
Senior analyst receives a structured evidence brief and approves or denies.

**Interrupt point 2 — Ambiguous evidence:**
```
interrupt_before: "final_decision"
Condition: 0.4 < fraud_score < 0.7
```
Human investigator reviews the conflicting signals and adds context before the system decides.

**Evidence brief handed to human (DISP-002):**
```
DISPUTE: DISP-002 | Amount: $1,200.00 | Merchant: Apple Store Online

EVIDENCE:
  ✓ Merchant has 0 prior fraud disputes
  ✓ Customer has 6 prior Apple Store purchases (avg $340)
  ⚠ This charge is 3.5x the customer's average Apple purchase
  ⚠ Charge originated from an unrecognized device
  ✓ Customer's home IP matches device location

FRAUD SCORE: 0.61 (AMBIGUOUS)
RECOMMENDATION: Human review required (amount $1,200 ≥ $500 threshold)

ACTION REQUIRED: approve_credit | deny | request_more_info
```

**Demo value:** The most powerful moment in any demo — the system does 90% of the work, then hands a clean brief to the human for the 10% that requires judgment. Shows AI augmenting humans, not replacing them.

---

### Phase 3.3 — Streaming & Observability

**What it does:** Streams investigation progress in real time and provides full audit traceability via LangSmith.

**Terminal dashboard output:**
```
[DISP-001] ▶ Intake complete — TXN-88821 | $340.00 | Biscayne Grill Miami
[DISP-001] ▶ Merchant check — 47 prior disputes, fraud rate 34% [FLAGGED]
[DISP-001] ▶ Customer profile — zero FL transactions in 90 days [FLAGGED]
[DISP-001] ▶ Fraud score: 0.91 — threshold exceeded
[DISP-001] ▶ Decision: AUTO_CREDIT
[DISP-001] ✓ $340.00 credited to CUST-4821 | elapsed: 4.2s
```

**LangSmith tracing:**
- Every tool call, every agent decision, every state transition is logged
- Critical for regulated industries — provides the audit trail for *why* a credit or denial decision was made
- Enables debugging when a model makes a wrong call

---

### Phase 3.4 — Error Handling & Retries

**What it does:** Makes the system production-resilient — retries flaky integrations, validates data at boundaries, fails gracefully.

**Pydantic models at intake:**
```python
class Transaction(BaseModel):
    id: str
    amount: float = Field(gt=0)
    merchant: str
    merchant_id: str
    date: date
    location: str

class FraudDecision(BaseModel):
    dispute_id: str
    fraud_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["auto_credit", "deny", "human_review"]
    resolution_amount: float
    evidence: list[str]
```

**Retry logic:**
- Merchant history API: retry up to 3× with exponential backoff (these integrations are flaky)
- If LLM returns a fraud score outside `[0.0, 1.0]`, catch the validation error and re-prompt
- If all external checks fail: route to `human_review` — never make an auto-decision on incomplete evidence

---

## Tech Stack

| Layer | Choice |
|---|---|
| LLM | Claude (Anthropic) via `langchain-anthropic` |
| Orchestration | LangGraph |
| Short-term memory | LangGraph `SqliteSaver` |
| Long-term memory | ChromaDB |
| Observability | LangSmith |
| Data validation | Pydantic v2 |
| Runtime | Python 3.11+, `uv` for dependency management |

---

## Project Structure (target)

```
fraud-dispute-agent/
├── README.md
├── REQUIREMENTS.md          ← this file
├── pyproject.toml
├── data/
│   └── disputes.json        ← mock dispute scenarios
├── phase1_1_single_agent/
│   └── agent.py
├── phase1_2_state_machine/
│   └── graph.py
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
└── shared/
    ├── models.py             ← Pydantic models (Transaction, DisputeState, FraudDecision)
    ├── mock_data.py          ← dispute + customer + merchant fixtures
    └── tools.py              ← shared tool definitions
```

---

## The Demo Arc (10 minutes)

| Time | What the audience sees |
|---|---|
| 0–2 min | DISP-001: open-and-shut fraud. System resolves in 4 seconds. *"Today, this takes 10 days."* |
| 3–5 min | DISP-004: velocity attack. Parallel agents catch a card-cloning pattern that a sequential check would miss. |
| 6–8 min | DISP-002: system is uncertain. Pauses. Hands a clean evidence brief to a human. Analyst approves in 10 seconds. |
| 9–10 min | LangSmith trace: *"Here is every decision the system made, and why — full audit log."* |

---

## Key Talking Points

- **Speed:** Seconds vs. 7–10 business days for straightforward disputes
- **Scale:** One system handles thousands of disputes simultaneously
- **Trust:** Humans stay in the loop for high-value and ambiguous cases — AI augments, not replaces
- **Auditability:** Every decision is traceable — critical for financial services compliance
- **Learning:** The system gets smarter over time as merchant and customer patterns accumulate in memory
