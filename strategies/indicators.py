from __future__ import annotations

from typing import Iterable, List, Optional


def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    return sum(values[-period:]) / period


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    k = 2 / (period + 1)
    ema_val = values[-period]
    for value in values[-period + 1 :]:
        ema_val = value * k + ema_val * (1 - k)
    return ema_val


def atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / period


def vwap(prices: List[float], volumes: List[float]) -> Optional[float]:
    if not prices or not volumes or len(prices) != len(volumes):
        return None
    total_volume = sum(volumes)
    if total_volume == 0:
        return None
    return sum(p * v for p, v in zip(prices, volumes)) / total_volume


def bollinger_bands(values: List[float], period: int, std_mult: float) -> Optional[tuple[float, float, float]]:
    if len(values) < period or period <= 0:
        return None
    window = values[-period:]
    mean = sum(window) / period
    variance = sum((v - mean) ** 2 for v in window) / period
    std = variance ** 0.5
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    return lower, mean, upper


def rsi(values: List[float], period: int) -> Optional[float]:
    if len(values) < period + 1 or period <= 0:
        return None
    gains = []
    losses = []
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(abs(min(change, 0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
