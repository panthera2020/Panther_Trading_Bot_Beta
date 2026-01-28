from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from exchange.bybit_client import BybitClient, BybitConfig
from main import BotConfig, TradingBot


app = FastAPI()
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app")


class BotActionResponse(BaseModel):
    status: str


class StartBotRequest(BaseModel):
    strategies: list[str] | None = None
    test_trade: bool = True


api_key = os.getenv("BYBIT_API_KEY", "")
api_secret = os.getenv("BYBIT_API_SECRET", "")
if not api_key or not api_secret:
    raise RuntimeError("BYBIT_API_KEY/BYBIT_API_SECRET are required.")

exchange_client = BybitClient(
    BybitConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=os.getenv("BYBIT_TESTNET", "true").lower() in {"1", "true", "yes"},
        demo=os.getenv("BYBIT_DEMO", "true").lower() in {"1", "true", "yes"},
        category=os.getenv("BYBIT_CATEGORY", "linear"),
    )
)
bot = TradingBot(BotConfig(), exchange_client)


@app.post("/bot/start", response_model=BotActionResponse)
def start_bot(payload: StartBotRequest) -> BotActionResponse:
    strategies = payload.strategies or ["trend", "scalp", "candle3"]
    bot.start(strategies=strategies, run_test_trade=payload.test_trade)
    return BotActionResponse(status="ok")


@app.post("/bot/stop", response_model=BotActionResponse)
def stop_bot() -> BotActionResponse:
    bot.stop()
    return BotActionResponse(status="ok")


@app.post("/bot/pause", response_model=BotActionResponse)
def pause_bot() -> BotActionResponse:
    bot.pause()
    return BotActionResponse(status="ok")


@app.post("/bot/terminate", response_model=BotActionResponse)
def terminate_bot() -> BotActionResponse:
    bot.terminate()
    return BotActionResponse(status="ok")


@app.get("/bot/status")
def bot_status() -> dict:
    status = bot.status()
    return {
        "state": status.state.value,
        "mode": status.mode.value,
        "daily_volume": status.daily_volume,
        "daily_target": status.daily_target,
        "monthly_volume": status.monthly_volume,
        "strategy_volume": status.strategy_volume,
        "open_positions": status.open_positions,
        "last_error": status.last_error,
        "balance": status.balance,
        "trade_stats": status.trade_stats,
        "open_trades": status.open_trades,
        "closed_trades": status.closed_trades,
    }
