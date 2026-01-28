from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Position:
    symbol: str
    strategy_id: str
    side: str
    size: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    opened_at: datetime


@dataclass
class TradeRecord:
    symbol: str
    strategy_id: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    pnl: float


class PositionManager:
    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}
        self._closed_trades: List[TradeRecord] = []

    def _key(self, symbol: str, strategy_id: str) -> str:
        return f"{symbol}:{strategy_id}"

    def has_open_position(self, symbol: str, strategy_id: str) -> bool:
        return self._key(symbol, strategy_id) in self._positions

    def open_position(self, position: Position) -> None:
        self._positions[self._key(position.symbol, position.strategy_id)] = position

    def close_position(self, symbol: str, strategy_id: str) -> None:
        self._positions.pop(self._key(symbol, strategy_id), None)

    def open_positions_count(self) -> int:
        return len(self._positions)

    def get_position(self, symbol: str, strategy_id: str) -> Optional[Position]:
        return self._positions.get(self._key(symbol, strategy_id))

    def close_position_with_price(
        self,
        symbol: str,
        strategy_id: str,
        exit_price: float,
        closed_at: datetime,
    ) -> Optional[TradeRecord]:
        position = self._positions.pop(self._key(symbol, strategy_id), None)
        if not position:
            return None
        if position.side.upper() == "BUY":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        trade = TradeRecord(
            symbol=position.symbol,
            strategy_id=position.strategy_id,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            opened_at=position.opened_at,
            closed_at=closed_at,
            pnl=pnl,
        )
        self._closed_trades.append(trade)
        return trade

    def open_positions(self) -> List[dict]:
        return [asdict(position) for position in self._positions.values()]

    def closed_trades(self, limit: int = 50) -> List[dict]:
        return [asdict(trade) for trade in self._closed_trades[-limit:]]

    def trade_stats(self) -> dict:
        total = len(self._closed_trades)
        pnl = sum(trade.pnl for trade in self._closed_trades)
        wins = sum(1 for trade in self._closed_trades if trade.pnl > 0)
        win_rate = (wins / total) * 100 if total > 0 else 0.0
        per_strategy: Dict[str, dict] = {}
        for trade in self._closed_trades:
            stats = per_strategy.setdefault(
                trade.strategy_id,
                {"trades": 0, "wins": 0, "win_rate": 0.0, "pnl": 0.0},
            )
            stats["trades"] += 1
            if trade.pnl > 0:
                stats["wins"] += 1
            stats["pnl"] += trade.pnl
        for stats in per_strategy.values():
            trades = stats["trades"]
            stats["win_rate"] = round((stats["wins"] / trades) * 100, 2) if trades > 0 else 0.0
            stats["pnl"] = round(stats["pnl"], 2)
        return {
            "trades": total,
            "wins": wins,
            "win_rate": round(win_rate, 2),
            "pnl": round(pnl, 2),
            "per_strategy": per_strategy,
        }
