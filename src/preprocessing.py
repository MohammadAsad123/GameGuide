"""
GameGuide — NLP Preprocessing Module
===================================
Standard text normalization and currency cleaning for user queries.
"""

import re
from typing import Optional


def preprocess(q: Optional[str]) -> str:
    """
    Normalize user query:
    1. Lowercase and strip whitespace.
    2. Convert Rs/rupees/inr expressions into standardized ₹<amount>.
    3. Retain alphanumeric, spaces, currency symbols, comparison signs, and punctuation.
    4. Collapse multiple spaces.
    """
    if q is None:
        return ""
    q = str(q).lower().strip()
    q = re.sub(r"(rs\.?|rupees?|inr)\s*(\d+)", r"₹\2", q)
    q = re.sub(r"(\d+)\s*(rs\.?|rupees?|inr)", r"₹\1", q)
    q = re.sub(r"[^\w\s₹$<>\.\-']", " ", q)
    return re.sub(r"\s+", " ", q).strip()
