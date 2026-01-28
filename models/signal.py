from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class TradeSignal:
    symbol: str
    strategy_id: str
    side: Side
    timestamp: datetime
    price: float
    stop_loss: float
    take_profit: Optional[float]
    size: float
    reason: str = ""
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)
