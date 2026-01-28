from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.03
    max_consecutive_losses: int = 3
    max_orders_per_hour: int = 20
    risk_per_trade_pct: float = 0.005


class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self._day_key = self._day_id(datetime.now(timezone.utc))
        self._equity_start = 0.0
        self._realized_pnl = 0.0
        self._consecutive_losses = 0
        self._orders_this_hour = 0
        self._hour_key = self._hour_id(datetime.now(timezone.utc))

    def _day_id(self, ts: datetime) -> str:
        return ts.strftime("%Y-%m-%d")

    def _hour_id(self, ts: datetime) -> str:
        return ts.strftime("%Y-%m-%d-%H")

    def start_day(self, equity: float, timestamp: datetime) -> None:
        self._day_key = self._day_id(timestamp)
        self._equity_start = equity
        self._realized_pnl = 0.0
        self._consecutive_losses = 0

    def register_order(self, timestamp: datetime) -> None:
        hour_key = self._hour_id(timestamp)
        if hour_key != self._hour_key:
            self._hour_key = hour_key
            self._orders_this_hour = 0
        self._orders_this_hour += 1

    def register_pnl(self, pnl: float) -> None:
        self._realized_pnl += pnl
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def can_trade(self, equity: float, timestamp: datetime) -> bool:
        if self._day_id(timestamp) != self._day_key:
            self.start_day(equity, timestamp)
        if self._equity_start <= 0:
            self._equity_start = equity
        max_loss = self._equity_start * self.config.max_daily_loss_pct
        if -self._realized_pnl >= max_loss:
            return False
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return False
        hour_key = self._hour_id(timestamp)
        if hour_key != self._hour_key:
            self._hour_key = hour_key
            self._orders_this_hour = 0
        if self._orders_this_hour >= self.config.max_orders_per_hour:
            return False
        return True
