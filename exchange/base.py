from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from exchange.types import OrderResult


class ExchangeClient(Protocol):
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult: ...

    def close_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult: ...

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> List[Dict[str, Any]]: ...

    def get_balance(self) -> Dict[str, float]: ...

    def get_last_price(self, symbol: str) -> float: ...
