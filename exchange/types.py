from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderResult:
    order_id: str
    status: str
    filled: float
    average_price: Optional[float]
