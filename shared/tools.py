"""
Shared LangChain tool definitions used across all phases.

Each tool is a pure function decorated with @tool. The docstring is
what the LLM sees when deciding whether to call the tool — write it
like an API contract, not a comment.

In a real system these would call external services. Here they query
the mock data module. The tool interface stays identical — only the
implementation changes at deployment time.
"""

import asyncio
from typing import Optional

from langchain_core.tools import tool

from shared.mock_data import (
    CUSTOMER_PROFILES,
    MERCHANT_PROFILES,
    TRANSACTIONS,
    VELOCITY_CHARGES,
)


@tool
def get_transaction(transaction_id: str) -> dict:
    """
    Fetch full transaction details for a given transaction ID.

    Returns merchant name, merchant ID, amount (USD), date, location,
    and device ID. Returns an error dict if the transaction is not found.
    """
    txn = TRANSACTIONS.get(transaction_id)
    if not txn:
        return {"error": f"Transaction {transaction_id} not found"}
    return txn


@tool
def check_merchant_fraud_history(merchant_id: str) -> dict:
    """
    Retrieve fraud and dispute history for a merchant.

    Returns:
    - prior_dispute_count: total disputes filed against this merchant
    - fraud_rate_pct: percentage of transactions that were confirmed fraud
    - known_skimmer: whether a card skimmer device has been reported at this merchant
    - flagged_categories: list of risk flags (e.g. 'high_dispute_volume', 'card_skimmer')

    A fraud_rate_pct > 10% or known_skimmer=True should significantly increase fraud score.
    """
    profile = MERCHANT_PROFILES.get(merchant_id)
    if not profile:
        return {"error": f"Merchant {merchant_id} not found", "merchant_id": merchant_id}
    return profile


@tool
def get_customer_spend_history(customer_id: str, days: int = 90) -> dict:
    """
    Retrieve a customer's recent spend history and profile.

    Returns:
    - home_city: customer's registered home location
    - recent_transactions: list of recent transactions (merchant, amount, location, date)
    - known_devices: list of device IDs the customer has previously used
    - avg_transaction_amount: customer's average transaction size

    Use this to:
    - Check if the disputed merchant's location matches anywhere the customer has been
    - Verify if the transaction device is recognised
    - Identify if the disputed amount is unusually large vs. their average
    """
    profile = CUSTOMER_PROFILES.get(customer_id)
    if not profile:
        return {"error": f"Customer {customer_id} not found", "customer_id": customer_id}
    return profile


@tool
def check_velocity_pattern(customer_id: str) -> dict:
    """
    Detect rapid sequential charges that indicate card cloning / velocity attacks.

    Returns:
    - charges: list of recent charges from the same merchant within a short window
    - is_velocity_attack: True if 3+ charges from the same merchant within 2 hours
    - time_span_minutes: time between first and last charge in the group
    - total_amount: sum of all charges in the group

    A velocity attack (is_velocity_attack=True) should push fraud_score to >= 0.9.
    """
    charges = VELOCITY_CHARGES.get(customer_id, [])
    if not charges:
        return {
            "customer_id": customer_id,
            "charges": [],
            "is_velocity_attack": False,
            "time_span_minutes": 0,
            "total_amount": 0.0,
        }

    # Calculate time span
    from datetime import datetime
    timestamps = [datetime.fromisoformat(c["timestamp"]) for c in charges]
    span_minutes = (max(timestamps) - min(timestamps)).seconds // 60

    is_attack = len(charges) >= 3 and span_minutes <= 120

    return {
        "customer_id": customer_id,
        "charges": charges,
        "is_velocity_attack": is_attack,
        "charge_count": len(charges),
        "time_span_minutes": span_minutes,
        "total_amount": sum(c["amount"] for c in charges),
        "merchant_id": charges[0]["merchant_id"] if charges else None,
    }


# Convenience: all tools as a list for binding to LLM
ALL_TOOLS = [
    get_transaction,
    check_merchant_fraud_history,
    get_customer_spend_history,
    check_velocity_pattern,
]
