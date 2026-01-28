from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict


@dataclass
class VolumeConfig:
    monthly_target: float
    trading_days: int = 30
    strategy_allocations: Dict[str, float] = field(default_factory=dict)


class VolumeManager:
    def __init__(self, config: VolumeConfig):
        self.config = config
        self.daily_volume: float = 0.0
        self.monthly_volume: float = 0.0
        self.strategy_volume: Dict[str, float] = {k: 0.0 for k in config.strategy_allocations}
        self._current_day = self._day_key(datetime.now(timezone.utc))
        self._current_month = self._month_key(datetime.now(timezone.utc))

    @property
    def daily_target(self) -> float:
        return self.config.monthly_target / max(self.config.trading_days, 1)

    def _day_key(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    def _month_key(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m")

    def _roll_if_needed(self, dt: datetime) -> None:
        day_key = self._day_key(dt)
        month_key = self._month_key(dt)
        if day_key != self._current_day:
            self.daily_volume = 0.0
            self._current_day = day_key
        if month_key != self._current_month:
            self.monthly_volume = 0.0
            self.strategy_volume = {k: 0.0 for k in self.strategy_volume}
            self._current_month = month_key

    def register_trade(self, strategy_id: str, notional: float, timestamp: datetime) -> None:
        self._roll_if_needed(timestamp)
        self.daily_volume += notional
        self.monthly_volume += notional
        self.strategy_volume.setdefault(strategy_id, 0.0)
        self.strategy_volume[strategy_id] += notional

    def remaining_daily_volume(self, timestamp: datetime) -> float:
        self._roll_if_needed(timestamp)
        return max(self.daily_target - self.daily_volume, 0.0)

    def strategy_remaining(self, strategy_id: str, timestamp: datetime) -> float:
        self._roll_if_needed(timestamp)
        allocation = self.config.strategy_allocations.get(strategy_id, 1.0)
        target = self.daily_target * allocation
        return max(target - self.strategy_volume.get(strategy_id, 0.0), 0.0)

    def compute_size(
        self,
        strategy_id: str,
        risk_pct: float,
        equity: float,
        atr: float,
        k: float,
        expected_trades_left: int,
        price: float,
        timestamp: datetime,
    ) -> float:
        if atr <= 0 or k <= 0 or price <= 0:
            return 0.0
        base_size = (risk_pct * equity) / (atr * k)
        remaining_volume = self.strategy_remaining(strategy_id, timestamp)
        if expected_trades_left <= 0:
            return 0.0
        max_size = (remaining_volume / expected_trades_left) / price
        return min(base_size, max_size)
