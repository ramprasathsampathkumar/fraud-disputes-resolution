"""
Mock fixtures for all 4 demo dispute scenarios.

In a real system these would come from a transaction DB, fraud data warehouse,
and customer profile service. We isolate them here so every phase can import
the same fixtures without duplicating data.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Dispute intake payloads  (what arrives at the system entry point)
# ---------------------------------------------------------------------------

DISPUTES: dict[str, dict] = {
    "DISP-001": {
        "dispute_id": "DISP-001",
        "customer_id": "CUST-4821",
        "transaction_id": "TXN-88821",
    },
    "DISP-002": {
        "dispute_id": "DISP-002",
        "customer_id": "CUST-1143",
        "transaction_id": "TXN-55302",
    },
    "DISP-003": {
        "dispute_id": "DISP-003",
        "customer_id": "CUST-7762",
        "transaction_id": "TXN-11294",
    },
    "DISP-004": {
        "dispute_id": "DISP-004",
        "customer_id": "CUST-3391",
        "transaction_id": "TXN-77401",  # first of 5 rapid charges
    },
}

# ---------------------------------------------------------------------------
# Transaction records  (what the `get_transaction` tool returns)
# ---------------------------------------------------------------------------

TRANSACTIONS: dict[str, dict] = {
    "TXN-88821": {
        "id": "TXN-88821",
        "amount": 340.00,
        "merchant": "Biscayne Grill",
        "merchant_id": "MCH-0042",
        "date": "2026-04-10",
        "location": "Miami, FL",
        "device_id": "DEV-UNKNOWN-9921",
    },
    "TXN-55302": {
        "id": "TXN-55302",
        "amount": 1200.00,
        "merchant": "Apple Store Online",
        "merchant_id": "MCH-0001",
        "date": "2026-04-11",
        "location": "Online",
        "device_id": "DEV-UNKNOWN-3341",  # unrecognised device
    },
    "TXN-11294": {
        "id": "TXN-11294",
        "amount": 18.50,
        "merchant": "Netflix",
        "merchant_id": "MCH-0099",
        "date": "2026-04-12",
        "location": "Online",
        "device_id": "DEV-7762-HOME",
    },
    # DISP-004: 5 rapid charges — we expose them as a list under a group key
    "TXN-77401": {
        "id": "TXN-77401",
        "amount": 340.00,
        "merchant": "Shell Gas Station",
        "merchant_id": "MCH-0512",
        "date": "2026-04-13",
        "location": "Houston, TX",
        "device_id": "DEV-UNKNOWN-0001",
    },
}

# The 5 rapid charges for DISP-004 — used by the velocity tool
VELOCITY_CHARGES: dict[str, list[dict]] = {
    "CUST-3391": [
        {"txn_id": "TXN-77401", "amount": 340.00, "merchant_id": "MCH-0512", "timestamp": "2026-04-13T14:01:00", "location": "Houston, TX"},
        {"txn_id": "TXN-77402", "amount": 340.00, "merchant_id": "MCH-0512", "timestamp": "2026-04-13T14:22:00", "location": "Houston, TX"},
        {"txn_id": "TXN-77403", "amount": 340.00, "merchant_id": "MCH-0512", "timestamp": "2026-04-13T14:45:00", "location": "Houston, TX"},
        {"txn_id": "TXN-77404", "amount": 340.00, "merchant_id": "MCH-0512", "timestamp": "2026-04-13T15:03:00", "location": "Houston, TX"},
        {"txn_id": "TXN-77405", "amount": 340.00, "merchant_id": "MCH-0512", "timestamp": "2026-04-13T15:19:00", "location": "Houston, TX"},
    ]
}

# ---------------------------------------------------------------------------
# Merchant risk profiles  (what `check_merchant_fraud_history` returns)
# ---------------------------------------------------------------------------

MERCHANT_PROFILES: dict[str, dict] = {
    "MCH-0042": {
        "merchant_id": "MCH-0042",
        "merchant_name": "Biscayne Grill",
        "prior_dispute_count": 47,
        "fraud_rate_pct": 34.2,
        "known_skimmer": False,
        "flagged_categories": ["high_dispute_volume", "tourist_area"],
    },
    "MCH-0001": {
        "merchant_id": "MCH-0001",
        "merchant_name": "Apple Store Online",
        "prior_dispute_count": 3,
        "fraud_rate_pct": 0.8,
        "known_skimmer": False,
        "flagged_categories": [],
    },
    "MCH-0099": {
        "merchant_id": "MCH-0099",
        "merchant_name": "Netflix",
        "prior_dispute_count": 1,
        "fraud_rate_pct": 0.2,
        "known_skimmer": False,
        "flagged_categories": [],
    },
    "MCH-0512": {
        "merchant_id": "MCH-0512",
        "merchant_name": "Shell Gas Station",
        "prior_dispute_count": 12,
        "fraud_rate_pct": 8.1,
        "known_skimmer": True,  # <-- skimmer device previously reported
        "flagged_categories": ["card_skimmer", "gas_station"],
    },
}

# ---------------------------------------------------------------------------
# Customer profiles  (what `get_customer_spend_history` returns)
# ---------------------------------------------------------------------------

CUSTOMER_PROFILES: dict[str, dict] = {
    "CUST-4821": {
        "customer_id": "CUST-4821",
        "name": "Sarah Chen",
        "home_city": "Seattle, WA",
        "known_devices": ["DEV-4821-IPHONE", "DEV-4821-MACBOOK"],
        "recent_transactions": [
            {"merchant": "Whole Foods", "amount": 87.40, "location": "Seattle, WA", "date": "2026-04-09"},
            {"merchant": "Starbucks", "amount": 6.50, "location": "Seattle, WA", "date": "2026-04-08"},
            {"merchant": "Amazon", "amount": 134.99, "location": "Online", "date": "2026-04-07"},
            {"merchant": "Safeway", "amount": 62.10, "location": "Seattle, WA", "date": "2026-04-05"},
            {"merchant": "United Airlines", "amount": 480.00, "location": "Online", "date": "2026-03-20"},
        ],
        # No Florida transactions in past 90 days
        "avg_transaction_amount": 154.20,
    },
    "CUST-1143": {
        "customer_id": "CUST-1143",
        "name": "Marcus Johnson",
        "home_city": "Austin, TX",
        "known_devices": ["DEV-1143-IPHONE", "DEV-1143-IPAD"],
        "recent_transactions": [
            {"merchant": "Apple Store Online", "amount": 399.00, "location": "Online", "date": "2026-02-14"},
            {"merchant": "Apple Store Online", "amount": 279.00, "location": "Online", "date": "2025-12-20"},
            {"merchant": "Apple Store", "amount": 449.00, "location": "Austin, TX", "date": "2025-11-03"},
            {"merchant": "Apple Store Online", "amount": 199.00, "location": "Online", "date": "2025-09-15"},
            {"merchant": "Best Buy", "amount": 899.00, "location": "Austin, TX", "date": "2025-08-22"},
        ],
        # Regular Apple buyer — but avg Apple spend is ~$331, this charge is $1,200
        "avg_transaction_amount": 445.00,
    },
    "CUST-7762": {
        "customer_id": "CUST-7762",
        "name": "Priya Patel",
        "home_city": "Chicago, IL",
        "known_devices": ["DEV-7762-HOME"],
        "recent_transactions": [
            {"merchant": "Netflix", "amount": 18.50, "location": "Online", "date": "2026-03-12"},
            {"merchant": "Netflix", "amount": 18.50, "location": "Online", "date": "2026-02-12"},
            {"merchant": "Netflix", "amount": 18.50, "location": "Online", "date": "2026-01-12"},
            {"merchant": "Trader Joe's", "amount": 55.30, "location": "Chicago, IL", "date": "2026-04-11"},
        ],
        # Active Netflix subscriber — recurring monthly charge
        "avg_transaction_amount": 47.70,
    },
    "CUST-3391": {
        "customer_id": "CUST-3391",
        "name": "David Kim",
        "home_city": "Portland, OR",
        "known_devices": ["DEV-3391-ANDROID"],
        "recent_transactions": [
            {"merchant": "Fred Meyer", "amount": 94.20, "location": "Portland, OR", "date": "2026-04-12"},
            {"merchant": "Shell Gas", "amount": 65.00, "location": "Portland, OR", "date": "2026-04-10"},
            {"merchant": "Nike", "amount": 120.00, "location": "Portland, OR", "date": "2026-04-08"},
        ],
        # Never been to Texas — 5 × $340 at Houston Shell in 2 hours is anomalous
        "avg_transaction_amount": 93.07,
    },
}
