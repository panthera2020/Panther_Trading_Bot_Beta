from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from exchange.base import ExchangeClient
from execution.position_manager import Position, PositionManager
from execution.risk_manager import RiskManager
from execution.volume_manager import VolumeManager
from models.signal import TradeSignal


@dataclass
class OrderManagerConfig:
    symbol: str
    order_type: str = "market"


class OrderManager:
    def __init__(
        self,
        client: ExchangeClient,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        volume_manager: VolumeManager,
        config: OrderManagerConfig,
    ):
        self.client = client
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.volume_manager = volume_manager
        self.config = config

    def execute_signal(self, signal: TradeSignal, timestamp: datetime) -> Optional[str]:
        if self.position_manager.has_open_position(signal.symbol, signal.strategy_id):
            return None
        self.risk_manager.register_order(timestamp)
        result = self.client.create_order(
            symbol=signal.symbol,
            side=signal.side.value.lower(),
            order_type=self.config.order_type,
            amount=signal.size,
            price=None,
        )
        if result.status in {"open", "closed"}:
            position = Position(
                symbol=signal.symbol,
                strategy_id=signal.strategy_id,
                side=signal.side.value,
                size=signal.size,
                entry_price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                opened_at=signal.timestamp,
            )
            self.position_manager.open_position(position)
            notional = signal.price * signal.size
            self.volume_manager.register_trade(signal.strategy_id, notional, timestamp)
            return result.order_id
        return None

    def close_position(self, symbol: str, strategy_id: str, exit_price: float, timestamp: datetime) -> None:
        position = self.position_manager.get_position(symbol, strategy_id)
        if not position:
            return
        close_side = "sell" if position.side.upper() == "BUY" else "buy"
        self.client.close_position(
            symbol=symbol,
            side=close_side,
            amount=position.size,
        )
        trade = self.position_manager.close_position_with_price(symbol, strategy_id, exit_price, timestamp)
        if trade:
            self.risk_manager.register_pnl(trade.pnl)
