"""Output validation that goes beyond shape: is the value plausible, fresh and consistent?

A price of 0.9912 is a perfectly valid float. It is also wrong if the market quotes in percent of par.
These checks run on every vendor response the agent wants to act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Settings
from .models import PriceQuote
from .observability import Recorder


@dataclass
class GuardResult:
    implausible: list[str] = field(default_factory=list)  # wrong value: reject, count against the feed
    stale: list[str] = field(default_factory=list)  # right value, too old: a human must source a fresh one

    @property
    def ok(self) -> bool:
        return not (self.implausible or self.stale)

    @property
    def problems(self) -> list[str]:
        return self.implausible + self.stale


def check_price(q: PriceQuote, s: Settings, now: datetime | None = None) -> GuardResult:
    r = GuardResult()
    if not (s.price_min <= q.clean_price <= s.price_max):
        r.implausible.append(
            f"{q.source} price {q.clean_price} outside plausible range {s.price_min}-{s.price_max} "
            "(percent of par); possible vendor format change"
        )
    now = now or datetime.now(timezone.utc)
    age_h = (now - q.as_of).total_seconds() / 3600
    if age_h > s.max_price_age_hours:
        r.stale.append(f"{q.source} price is {age_h:.0f}h old, limit {s.max_price_age_hours:.0f}h")
    return r


class CircuitBreaker:
    """Stops writes for a feed when too many of its recent responses failed validation."""

    def __init__(self, recorder: Recorder, s: Settings):
        self.recorder = recorder
        self.s = s

    def record(self, feed: str, ok: bool, run_id: str) -> None:
        self.recorder.feed_event(feed, ok, run_id)

    def is_open(self, feed: str) -> bool:
        events = self.recorder.recent_feed_events(feed, self.s.breaker_window)
        if len(events) < self.s.breaker_min_events:
            return False
        failure_rate = events.count(False) / len(events)
        return failure_rate >= self.s.breaker_failure_rate
