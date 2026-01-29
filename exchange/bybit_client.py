from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional

from pybit.unified_trading import HTTP

from exchange.types import OrderResult


@dataclass
class BybitConfig:
    api_key: str
    api_secret: str
    testnet: bool = True
    demo: bool = True
    category: str = "linear"
    recv_window: int = 10000
    timeout: int = 20


class BybitClient:
    """
    Official Bybit SDK (pybit) client.
    Uses Unified Trading HTTP with demo/testnet support.
    """

    def __init__(self, config: BybitConfig):
        self.config = config
        self._session = HTTP(
            testnet=config.testnet,
            demo=config.demo,
            api_key=config.api_key,
            api_secret=config.api_secret,
            recv_window=config.recv_window,
            timeout=config.timeout,
        )
        self._lot_filters: Dict[str, Dict[str, float]] = {}
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_ts: float = 0.0

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        payload: Dict[str, Any] = {
            "category": self.config.category,
            "symbol": symbol,
            "side": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Market" if order_type.lower() == "market" else "Limit",
            "qty": str(amount),
        }
        if price is not None:
            payload["price"] = str(price)
        if params:
            payload.update(params)
        response = self._session.place_order(**payload)
        result = response.get("result", {})
        return OrderResult(
            order_id=str(result.get("orderId", "")),
            status=response.get("retMsg", "unknown"),
            filled=float(result.get("orderQty", 0.0) or 0.0),
            average_price=None,
        )

    def close_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        payload: Dict[str, Any] = {
            "category": self.config.category,
            "symbol": symbol,
            "side": "Buy" if side.lower() == "buy" else "Sell",
            "orderType": "Market",
            "qty": str(amount),
            "reduceOnly": True,
        }
        if params:
            payload.update(params)
        response = self._session.place_order(**payload)
        result = response.get("result", {})
        return OrderResult(
            order_id=str(result.get("orderId", "")),
            status=response.get("retMsg", "unknown"),
            filled=float(result.get("orderQty", 0.0) or 0.0),
            average_price=None,
        )

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> List[Dict[str, Any]]:
        interval_map = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "D",
        }
        interval = interval_map.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        response = self._session.get_kline(
            category=self.config.category,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        rows = response.get("result", {}).get("list", [])
        candles = []
        for row in rows:
            candles.append(
                {
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        return list(reversed(candles))

    def get_balance(self) -> Dict[str, float]:
        response = self._session.get_wallet_balance(accountType="UNIFIED")
        result = response.get("result", {})
        balances = result.get("list", [])
        if not balances:
            return {"total_equity": 0.0, "available_balance": 0.0}
        account = balances[0]
        return {
            "total_equity": float(account.get("totalEquity", 0.0) or 0.0),
            "available_balance": float(account.get("totalAvailableBalance", 0.0) or 0.0),
        }

    def get_last_price(self, symbol: str) -> float:
        response = self._session.get_tickers(category=self.config.category, symbol=symbol)
        tickers = response.get("result", {}).get("list", [])
        if not tickers:
            return 0.0
        return float(tickers[0].get("lastPrice", 0.0) or 0.0)

    def normalize_qty(self, symbol: str, qty: float) -> float:
        if qty <= 0:
            return 0.0
        filters = self._get_lot_filters(symbol)
        min_qty = filters.get("min_qty", 0.0)
        step = filters.get("qty_step", 0.0)
        if qty < min_qty:
            return 0.0
        if step > 0:
            qty = math.floor(qty / step) * step
        return float(qty)

    def _get_lot_filters(self, symbol: str) -> Dict[str, float]:
        if symbol in self._lot_filters:
            return self._lot_filters[symbol]
        response = self._session.get_instruments_info(category=self.config.category, symbol=symbol)
        items = response.get("result", {}).get("list", [])
        if not items:
            self._lot_filters[symbol] = {"min_qty": 0.0, "qty_step": 0.0}
            return self._lot_filters[symbol]
        lot = items[0].get("lotSizeFilter", {})
        min_qty = float(lot.get("minOrderQty", 0.0) or 0.0)
        step = float(lot.get("qtyStep", 0.0) or 0.0)
        self._lot_filters[symbol] = {"min_qty": min_qty, "qty_step": step}
        return self._lot_filters[symbol]

    def get_exchange_stats(self, symbols: List[str]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        if self._stats_cache and (now.timestamp() - self._stats_cache_ts) < 15:
            return self._stats_cache

        symbol = symbols[0] if symbols else "BTCUSDT"
        start_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        start_week = now - timedelta(days=7)

        try:
            executions = self._fetch_executions(symbol, start_week, now)
            volume_daily = self._sum_volume(executions, start_day)
            volume_weekly = self._sum_volume(executions, start_week)

            closed_pnl = self._fetch_closed_pnl(symbol, start_week, now)
            trade_stats = self._compute_trade_stats(closed_pnl, start_week)
            closed_trades = self._map_closed_trades(closed_pnl, limit=10)
            open_trades = self._fetch_open_positions(symbol)
        except Exception:
            if self._stats_cache:
                return self._stats_cache
            volume_daily = 0.0
            volume_weekly = 0.0
            trade_stats = {"trades": 0, "wins": 0, "win_rate": 0.0, "pnl": 0.0}
            closed_trades = []
            open_trades = []

        stats = {
            "volume": {
                "daily": round(volume_daily, 2),
                "weekly": round(volume_weekly, 2),
            },
            "trade_stats": trade_stats,
            "open_trades": open_trades,
            "closed_trades": closed_trades,
        }
        self._stats_cache = stats
        self._stats_cache_ts = now.timestamp()
        return stats

    def _fetch_executions(self, symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + timedelta(days=7), end)
            results.extend(self._fetch_executions_window(symbol, window_start, window_end))
            window_start = window_end
        return results

    def _fetch_executions_window(self, symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        results: List[Dict[str, Any]] = []
        cursor = None
        for _ in range(5):
            payload: Dict[str, Any] = {
                "category": self.config.category,
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 200,
            }
            if cursor:
                payload["cursor"] = cursor
            response = self._session.get_executions(**payload)
            items = response.get("result", {}).get("list", [])
            results.extend(items)
            cursor = response.get("result", {}).get("nextPageCursor")
            if not cursor:
                break
        return results

    def _fetch_closed_pnl(self, symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + timedelta(days=7), end)
            results.extend(self._fetch_closed_pnl_window(symbol, window_start, window_end))
            window_start = window_end
        return results

    def _fetch_closed_pnl_window(self, symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        results: List[Dict[str, Any]] = []
        cursor = None
        for _ in range(5):
            payload: Dict[str, Any] = {
                "category": self.config.category,
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 200,
            }
            if cursor:
                payload["cursor"] = cursor
            response = self._session.get_closed_pnl(**payload)
            items = response.get("result", {}).get("list", [])
            results.extend(items)
            cursor = response.get("result", {}).get("nextPageCursor")
            if not cursor:
                break
        return results

    def _fetch_open_positions(self, symbol: str) -> List[Dict[str, Any]]:
        response = self._session.get_positions(category=self.config.category, symbol=symbol)
        positions = response.get("result", {}).get("list", [])
        mapped = []
        for pos in positions:
            size = float(pos.get("size", 0.0) or 0.0)
            if size == 0:
                continue
            mapped.append(
                {
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side"),
                    "size": size,
                    "entry_price": float(pos.get("avgPrice", 0.0) or 0.0),
                    "unrealized_pnl": float(pos.get("unrealisedPnl", 0.0) or 0.0),
                    "updated_at": pos.get("updatedTime"),
                }
            )
        return mapped

    def _sum_volume(self, executions: List[Dict[str, Any]], start: datetime) -> float:
        start_ms = int(start.timestamp() * 1000)
        volume = 0.0
        for exec_item in executions:
            exec_time = int(exec_item.get("execTime", 0) or 0)
            if exec_time < start_ms:
                continue
            price = float(exec_item.get("execPrice", 0.0) or 0.0)
            qty = float(exec_item.get("execQty", 0.0) or 0.0)
            volume += price * qty
        return volume

    def _compute_trade_stats(self, closed_pnl: List[Dict[str, Any]], start: datetime) -> Dict[str, Any]:
        start_ms = int(start.timestamp() * 1000)
        trades = []
        for item in closed_pnl:
            created = int(item.get("createdTime", 0) or 0)
            if created < start_ms:
                continue
            trades.append(item)
        total = len(trades)
        pnl = sum(float(t.get("closedPnl", 0.0) or 0.0) for t in trades)
        wins = sum(1 for t in trades if float(t.get("closedPnl", 0.0) or 0.0) > 0)
        win_rate = (wins / total) * 100 if total else 0.0
        return {
            "trades": total,
            "wins": wins,
            "win_rate": round(win_rate, 2),
            "pnl": round(pnl, 2),
        }

    def _map_closed_trades(self, closed_pnl: List[Dict[str, Any]], limit: int = 50) -> List[Dict[str, Any]]:
        mapped = []
        for item in closed_pnl[:limit]:
            mapped.append(
                {
                    "symbol": item.get("symbol"),
                    "side": item.get("side"),
                    "qty": float(item.get("qty", 0.0) or 0.0),
                    "entry_price": float(item.get("avgEntryPrice", 0.0) or 0.0),
                    "exit_price": float(item.get("avgExitPrice", 0.0) or 0.0),
                    "pnl": float(item.get("closedPnl", 0.0) or 0.0),
                    "closed_at": item.get("createdTime"),
                }
            )
        return mapped
