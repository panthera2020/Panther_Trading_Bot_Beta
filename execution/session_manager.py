from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict


@dataclass(frozen=True)
class SessionPolicy:
    name: str
    start_hour: int
    end_hour: int
    strategy_size_mult: Dict[str, float]


class SessionManager:
    def __init__(self) -> None:
        self._sessions = [
            SessionPolicy(name="ASIA", start_hour=0, end_hour=8, strategy_size_mult={"scalp": 0.6, "trend": 0.3}),
            SessionPolicy(name="LONDON", start_hour=8, end_hour=16, strategy_size_mult={"scalp": 0.8, "trend": 0.6}),
            SessionPolicy(name="NY", start_hour=16, end_hour=24, strategy_size_mult={"scalp": 1.0, "trend": 1.0}),
        ]

    def current_session(self, timestamp: datetime | None = None) -> SessionPolicy:
        ts = timestamp or datetime.now(timezone.utc)
        hour = ts.hour
        for session in self._sessions:
            if session.start_hour <= hour < session.end_hour:
                return session
        return self._sessions[-1]

    def is_within_window(
        self,
        timestamp: datetime,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int,
        tz_offset_hours: int = 0,
    ) -> bool:
        tz = timezone(timedelta(hours=tz_offset_hours))
        local_ts = timestamp.astimezone(tz)
        start = local_ts.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end = local_ts.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        if end <= start:
            return local_ts >= start or local_ts <= end
        return start <= local_ts <= end
