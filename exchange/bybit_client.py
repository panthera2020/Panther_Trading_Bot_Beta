from __future__ import annotations

from dataclasses import dataclass
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
        )
        self._lot_filters: Dict[str, Dict[str, float]] = {}

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
