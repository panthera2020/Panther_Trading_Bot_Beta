from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class BotState(str, Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    PAUSED = "PAUSED"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"


class BotMode(str, Enum):
    IDLE = "IDLE"
    TEST_TRADE = "TEST_TRADE"
    SCANNING = "SCANNING"


@dataclass
class BotStatus:
    state: BotState
    mode: BotMode
    daily_volume: float
    daily_target: float
    monthly_volume: float
    strategy_volume: Dict[str, float] = field(default_factory=dict)
    open_positions: int = 0
    last_error: Optional[str] = None
    balance: Dict[str, float] = field(default_factory=dict)
    trade_stats: Dict[str, float] = field(default_factory=dict)
    open_trades: List[dict] = field(default_factory=list)
    closed_trades: List[dict] = field(default_factory=list)
