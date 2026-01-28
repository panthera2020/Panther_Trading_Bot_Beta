from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from models.signal import Side, TradeSignal
from strategies.indicators import atr, bollinger_bands, rsi, vwap


@dataclass
class MeanReversionConfig:
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    rsi_period: int = 14
    atr_k: float = 1.5
    max_holding_bars: int = 12
    use_rsi: bool = True


class MeanReversionStrategy:
    strategy_id = "scalp"

    def __init__(self, config: MeanReversionConfig):
        self.config = config

    def generate_signal(
        self,
        candles: List[Dict[str, float]],
        size: float,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[TradeSignal]:
        if len(candles) < max(self.config.bb_period, self.config.atr_period, self.config.rsi_period) + 2:
            return None

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        bands = bollinger_bands(closes, self.config.bb_period, self.config.bb_std)
        atr_val = atr(highs, lows, closes, self.config.atr_period)
        vwap_val = vwap(closes[-self.config.bb_period :], volumes[-self.config.bb_period :])
        rsi_val = rsi(closes, self.config.rsi_period) if self.config.use_rsi else None

        if bands is None or atr_val is None or vwap_val is None:
            return None

        lower, mid, upper = bands
        last_close = closes[-1]

        rsi_long_ok = (rsi_val is None) or (rsi_val < 30)
        rsi_short_ok = (rsi_val is None) or (rsi_val > 70)

        if last_close < lower and last_close < vwap_val and rsi_long_ok:
            stop = last_close - self.config.atr_k * atr_val
            take_profit = min(vwap_val, mid)
            return TradeSignal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                side=Side.BUY,
                timestamp=timestamp,
                price=last_close,
                stop_loss=stop,
                take_profit=take_profit,
                size=size,
                reason="mean_reversion_long",
                metadata={"max_holding_bars": self.config.max_holding_bars},
            )

        if last_close > upper and last_close > vwap_val and rsi_short_ok:
            stop = last_close + self.config.atr_k * atr_val
            take_profit = max(vwap_val, mid)
            return TradeSignal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                side=Side.SELL,
                timestamp=timestamp,
                price=last_close,
                stop_loss=stop,
                take_profit=take_profit,
                size=size,
                reason="mean_reversion_short",
                metadata={"max_holding_bars": self.config.max_holding_bars},
            )

        return None
