from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from models.signal import Side, TradeSignal
from strategies.indicators import atr, ema, sma


@dataclass
class TrendBreakoutConfig:
    lookback: int = 20
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    volume_sma: int = 20
    atr_k: float = 2.0
    trail_atr_k: float = 1.5
    min_ema_gap: float = 0.005


class TrendBreakoutStrategy:
    strategy_id = "trend"

    def __init__(self, config: TrendBreakoutConfig):
        self.config = config

    def generate_signal(
        self,
        candles: List[Dict[str, float]],
        size: float,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[TradeSignal]:
        if len(candles) < max(self.config.ema_slow, self.config.lookback) + 2:
            return None

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]

        ema_fast_val = ema(closes, self.config.ema_fast)
        ema_slow_val = ema(closes, self.config.ema_slow)
        atr_val = atr(highs, lows, closes, self.config.atr_period)
        vol_sma = sma(volumes, self.config.volume_sma)

        if ema_fast_val is None or ema_slow_val is None or atr_val is None:
            return None

        ema_gap = abs(ema_fast_val - ema_slow_val) / ema_slow_val
        if ema_gap < self.config.min_ema_gap:
            return None

        recent_high = max(highs[-self.config.lookback :])
        recent_low = min(lows[-self.config.lookback :])
        last_close = closes[-1]

        volume_ok = vol_sma is None or volumes[-1] > vol_sma

        if ema_fast_val > ema_slow_val and last_close > recent_high and volume_ok:
            stop = last_close - self.config.atr_k * atr_val
            take_profit = last_close + self.config.atr_k * atr_val
            return TradeSignal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                side=Side.BUY,
                timestamp=timestamp,
                price=last_close,
                stop_loss=stop,
                take_profit=take_profit,
                size=size,
                reason="trend_breakout_long",
                metadata={"trail_atr_k": self.config.trail_atr_k},
            )

        if ema_fast_val < ema_slow_val and last_close < recent_low and volume_ok:
            stop = last_close + self.config.atr_k * atr_val
            take_profit = last_close - self.config.atr_k * atr_val
            return TradeSignal(
                symbol=symbol,
                strategy_id=self.strategy_id,
                side=Side.SELL,
                timestamp=timestamp,
                price=last_close,
                stop_loss=stop,
                take_profit=take_profit,
                size=size,
                reason="trend_breakout_short",
                metadata={"trail_atr_k": self.config.trail_atr_k},
            )

        return None
