from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, Thread
from time import sleep
from typing import Dict, List, Optional

from exchange.base import ExchangeClient
from execution.order_manager import OrderManager, OrderManagerConfig
from execution.position_manager import Position, PositionManager
from execution.risk_manager import RiskConfig, RiskManager
from execution.session_manager import SessionManager
from execution.volume_manager import VolumeConfig, VolumeManager
from models.status import BotMode, BotState, BotStatus
from strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from strategies.strategy_c import StrategyC, StrategyCConfig
from strategies.trend_breakout import TrendBreakoutConfig, TrendBreakoutStrategy


@dataclass
class BotConfig:
    symbols: List[str] = None
    monthly_volume_target: float = 3_000_000.0
    trading_days: int = 30
    expected_trades_left: Dict[str, int] = None
    equity: float = 100_000.0
    test_trade_qty: float = 0.001
    test_trade_symbol: str = "BTCUSDT"
    poll_interval_seconds: int = 60


class TradingBot:
    def __init__(self, config: BotConfig, exchange_client: ExchangeClient):
        self.config = config
        self.exchange_client = exchange_client
        self.state = BotState.STOPPED
        self.session_manager = SessionManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(RiskConfig())
        self.volume_manager = VolumeManager(
            VolumeConfig(
                monthly_target=config.monthly_volume_target,
                trading_days=config.trading_days,
                strategy_allocations={"trend": 0.25, "scalp": 0.55, "candle3": 0.2},
            )
        )
        # BTC-only for now; add other symbols when ready.
        self.symbols = config.symbols or ["BTCUSDT"]
        self.order_manager = OrderManager(
            client=exchange_client,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            volume_manager=self.volume_manager,
            config=OrderManagerConfig(symbol=""),
        )
        self.trend_strategy = TrendBreakoutStrategy(TrendBreakoutConfig())
        self.scalp_strategy = MeanReversionStrategy(MeanReversionConfig())
        self.strategy_c = StrategyC(StrategyCConfig())
        self.last_error: Optional[str] = None
        self.expected_trades_left = config.expected_trades_left or {"trend": 2, "scalp": 20, "candle3": 30}
        self.enabled_strategies = {"trend", "scalp", "candle3"}
        self._strategies_enabled = True
        self._test_trade_in_progress = False
        self._mode = BotMode.IDLE
        self._stop_event = Event()
        self._loop_thread: Optional[Thread] = None

    def start(self, strategies: Optional[List[str]] = None, run_test_trade: bool = True) -> None:
        if self.state in {BotState.TERMINATED, BotState.ERROR}:
            return
        if strategies:
            self.enabled_strategies = set(strategies)
        else:
            self.enabled_strategies = {"trend", "scalp", "candle3"}
        if run_test_trade:
            self._strategies_enabled = False
            self._test_trade_in_progress = True
            self._mode = BotMode.TEST_TRADE
            Thread(target=self._run_test_trade, daemon=True).start()
        else:
            self._strategies_enabled = True
            self._mode = BotMode.SCANNING
        self.state = BotState.RUNNING
        self._start_loop()

    def stop(self) -> None:
        self.state = BotState.STOPPED
        self._mode = BotMode.IDLE
        self._stop_loop()

    def pause(self) -> None:
        if self.state == BotState.RUNNING:
            self.state = BotState.PAUSED
            self._mode = BotMode.IDLE

    def terminate(self) -> None:
        self.state = BotState.TERMINATED
        self._mode = BotMode.IDLE
        self._stop_loop()

    def status(self) -> BotStatus:
        balance = {}
        exchange_stats: dict = {}
        try:
            balance = self.exchange_client.get_balance()
            exchange_stats = self.exchange_client.get_exchange_stats(self.symbols)
        except Exception as exc:
            if "Retryable error occurred" not in str(exc):
                self.last_error = str(exc)
        daily_volume = self.volume_manager.daily_volume
        monthly_volume = self.volume_manager.monthly_volume
        exchange_volume: dict = {}
        trade_stats = self.position_manager.trade_stats()
        open_trades = self.position_manager.open_positions()
        closed_trades = self.position_manager.closed_trades()
        open_positions_count = self.position_manager.open_positions_count()
        if exchange_stats.get("volume"):
            exchange_volume = exchange_stats["volume"]
            daily_volume = exchange_stats["volume"].get("daily", daily_volume)
        if exchange_stats.get("trade_stats"):
            trade_stats = exchange_stats["trade_stats"]
        if exchange_stats.get("open_trades"):
            open_trades = exchange_stats["open_trades"]
            open_positions_count = len(open_trades)
        if exchange_stats.get("closed_trades"):
            closed_trades = exchange_stats["closed_trades"]
        return BotStatus(
            state=self.state,
            mode=self._mode,
            daily_volume=daily_volume,
            daily_target=self.volume_manager.daily_target,
            monthly_volume=monthly_volume,
            exchange_volume=exchange_volume,
            strategy_volume=self.volume_manager.strategy_volume,
            open_positions=open_positions_count,
            last_error=self.last_error,
            balance=balance,
            trade_stats=trade_stats,
            open_trades=open_trades,
            closed_trades=closed_trades,
        )

    def _size_for_strategy(self, strategy_id: str, atr_value: float, price: float, timestamp: datetime) -> float:
        session = self.session_manager.current_session(timestamp)
        size_mult = session.strategy_size_mult.get(strategy_id, 1.0)
        base_size = self.volume_manager.compute_size(
            strategy_id=strategy_id,
            risk_pct=self.risk_manager.config.risk_per_trade_pct,
            equity=self.config.equity,
            atr=atr_value,
            k=1.0,
            expected_trades_left=self.expected_trades_left.get(strategy_id, 1),
            price=price,
            timestamp=timestamp,
        )
        return base_size * size_mult

    def on_market_data(
        self,
        candles_1h: Dict[str, List[Dict[str, float]]] | List[Dict[str, float]],
        candles_5m: Dict[str, List[Dict[str, float]]] | List[Dict[str, float]],
        candles_3m: Dict[str, List[Dict[str, float]]] | List[Dict[str, float]],
        timestamp: Optional[datetime] = None,
    ) -> None:
        if self.state != BotState.RUNNING:
            return
        if self._test_trade_in_progress or not self._strategies_enabled:
            return

        ts = timestamp or datetime.now(timezone.utc)
        if not self.risk_manager.can_trade(self.config.equity, ts):
            return

        session = self.session_manager.current_session(ts)
        allow_trend = (
            "trend" in self.enabled_strategies
            and session.name in {"LONDON", "NY"}
            and session.strategy_size_mult.get("trend", 0.0) > 0
        )
        allow_scalp = (
            "scalp" in self.enabled_strategies
            and session.name in {"LONDON", "NY"}
            and session.strategy_size_mult.get("scalp", 0.0) > 0
        )
        allow_c = (
            "candle3" in self.enabled_strategies
            and session.name == "ASIA"
            and session.strategy_size_mult.get("scalp", 0.0) > 0
            and self.session_manager.is_within_window(ts, 1, 0, 15, 30, tz_offset_hours=1)
        )

        candles_1h_map = (
            candles_1h if isinstance(candles_1h, dict) else {self.config.test_trade_symbol: candles_1h}
        )
        candles_5m_map = (
            candles_5m if isinstance(candles_5m, dict) else {self.config.test_trade_symbol: candles_5m}
        )
        candles_3m_map = (
            candles_3m if isinstance(candles_3m, dict) else {self.config.test_trade_symbol: candles_3m}
        )

        for symbol in self.symbols:
            if allow_trend and not self.position_manager.has_open_position(symbol, "trend"):
                candles = candles_1h_map.get(symbol, [])
                atr_val = self._estimate_atr(candles)
                if atr_val:
                    price = candles[-1]["close"] if candles else 0.0
                    size = self._size_for_strategy("trend", atr_val, price, ts)
                    size = self.exchange_client.normalize_qty(symbol, size)
                    if size <= 0:
                        continue
                    signal = self.trend_strategy.generate_signal(candles, size, symbol, ts)
                    if signal:
                        try:
                            self.order_manager.execute_signal(signal, ts)
                        except Exception as exc:  # exchange errors should halt the bot
                            self.last_error = str(exc)
                            self.state = BotState.ERROR
                            return

            if allow_scalp and not self.position_manager.has_open_position(symbol, "scalp"):
                candles = candles_5m_map.get(symbol, [])
                atr_val = self._estimate_atr(candles)
                if atr_val:
                    price = candles[-1]["close"] if candles else 0.0
                    size = self._size_for_strategy("scalp", atr_val, price, ts)
                    size = self.exchange_client.normalize_qty(symbol, size)
                    if size <= 0:
                        continue
                    signal = self.scalp_strategy.generate_signal(candles, size, symbol, ts)
                    if signal:
                        try:
                            self.order_manager.execute_signal(signal, ts)
                        except Exception as exc:  # exchange errors should halt the bot
                            self.last_error = str(exc)
                            self.state = BotState.ERROR
                            return

            if allow_c and not self.position_manager.has_open_position(symbol, "candle3"):
                candles = candles_3m_map.get(symbol, [])
                atr_val = self._estimate_atr(candles)
                if atr_val:
                    price = candles[-1]["close"] if candles else 0.0
                    size = self._size_for_strategy("candle3", atr_val, price, ts)
                    size = self.exchange_client.normalize_qty(symbol, size)
                    if size <= 0:
                        continue
                    signal = self.strategy_c.generate_signal(candles, size, symbol, ts)
                    if signal:
                        try:
                            self.order_manager.execute_signal(signal, ts)
                            Thread(
                                target=self._monitor_strategy_c,
                                args=(symbol, "candle3", 10 * 3 * 60),
                                daemon=True,
                            ).start()
                        except Exception as exc:  # exchange errors should halt the bot
                            self.last_error = str(exc)
                            self.state = BotState.ERROR
                            return

    def _estimate_atr(self, candles: List[Dict[str, float]]) -> Optional[float]:
        if len(candles) < 15:
            return None
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        trs = []
        for i in range(-14, 0):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        return sum(trs) / 14

    def _run_test_trade(self) -> None:
        try:
            entry_price = self.exchange_client.get_last_price(self.config.test_trade_symbol)
            self.exchange_client.create_order(
                symbol=self.config.test_trade_symbol,
                side="buy",
                order_type="market",
                amount=self.config.test_trade_qty,
            )
            self.position_manager.open_position(
                Position(
                    symbol=self.config.test_trade_symbol,
                    strategy_id="test",
                    side="BUY",
                    size=self.config.test_trade_qty,
                    entry_price=entry_price,
                    stop_loss=entry_price * 0.99 if entry_price else 0.0,
                    take_profit=None,
                    opened_at=datetime.now(timezone.utc),
                )
            )
            sleep(5)
            exit_price = self.exchange_client.get_last_price(self.config.test_trade_symbol)
            self.exchange_client.close_position(
                symbol=self.config.test_trade_symbol,
                side="sell",
                amount=self.config.test_trade_qty,
            )
            trade = self.position_manager.close_position_with_price(
                self.config.test_trade_symbol, "test", exit_price or entry_price, datetime.now(timezone.utc)
            )
            if trade:
                self.risk_manager.register_pnl(trade.pnl)
        except Exception as exc:
            self.last_error = str(exc)
            self.state = BotState.ERROR
        finally:
            self._test_trade_in_progress = False
            self._strategies_enabled = True
            if self.state == BotState.RUNNING:
                self._mode = BotMode.SCANNING

    def _monitor_strategy_c(self, symbol: str, strategy_id: str, delay_seconds: int) -> None:
        """
        Priority exit order:
        1) Stop-loss hit
        2) Timer expiry
        3) Bot pause/terminate/stop
        """
        end_time = datetime.now(timezone.utc).timestamp() + delay_seconds
        while datetime.now(timezone.utc).timestamp() < end_time:
            if self.state != BotState.RUNNING:
                try:
                    exit_price = self.exchange_client.get_last_price(symbol)
                    self.order_manager.close_position(symbol, strategy_id, exit_price, datetime.now(timezone.utc))
                except Exception as exc:
                    self.last_error = str(exc)
                    self.state = BotState.ERROR
                return

            position = self.position_manager.get_position(symbol, strategy_id)
            if not position:
                return
            try:
                price = self.exchange_client.get_last_price(symbol)
                if position.side.upper() == "BUY" and price <= position.stop_loss:
                    self.order_manager.close_position(symbol, strategy_id, price, datetime.now(timezone.utc))
                    return
                if position.side.upper() == "SELL" and price >= position.stop_loss:
                    self.order_manager.close_position(symbol, strategy_id, price, datetime.now(timezone.utc))
                    return
            except Exception as exc:
                self.last_error = str(exc)
                self.state = BotState.ERROR
                return
            sleep(1)

        try:
            exit_price = self.exchange_client.get_last_price(symbol)
            self.order_manager.close_position(symbol, strategy_id, exit_price, datetime.now(timezone.utc))
        except Exception as exc:
            self.last_error = str(exc)
            self.state = BotState.ERROR

    def _start_loop(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self._loop_thread = Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _stop_loop(self) -> None:
        self._stop_event.set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.state == BotState.RUNNING and self._strategies_enabled and not self._test_trade_in_progress:
                try:
                    candles_1h = {s: self.exchange_client.fetch_ohlcv(s, "1h", 300) for s in self.symbols}
                    candles_5m = {s: self.exchange_client.fetch_ohlcv(s, "5m", 300) for s in self.symbols}
                    candles_3m = {s: self.exchange_client.fetch_ohlcv(s, "3m", 300) for s in self.symbols}
                    self._mode = BotMode.SCANNING
                    self.on_market_data(candles_1h, candles_5m, candles_3m, datetime.now(timezone.utc))
                except Exception as exc:
                    # Network/exchange hiccups: keep bot running and retry after 2 minutes.
                    self.last_error = str(exc)
                    self._mode = BotMode.IDLE
                    sleep(120)
                    continue
            sleep(self.config.poll_interval_seconds)
