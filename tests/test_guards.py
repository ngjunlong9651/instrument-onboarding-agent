from datetime import datetime, timedelta, timezone

from onboarding_agent.guards import CircuitBreaker, check_price
from onboarding_agent.models import PriceQuote
from onboarding_agent.observability import get_recorder

NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def q(px, age_h=1):
    return PriceQuote(isin="XS3333333336", source="secondary", clean_price=px, as_of=NOW - timedelta(hours=age_h))


def test_format_drift_is_caught_even_though_payload_is_valid(settings):
    r = check_price(q(0.9915), settings, now=NOW)
    assert r.implausible and "format change" in r.implausible[0]


def test_stale_price(settings):
    r = check_price(q(99.15, age_h=72), settings, now=NOW)
    assert r.stale and not r.implausible


def test_good_price(settings):
    assert check_price(q(99.15), settings, now=NOW).ok


def test_circuit_breaker_opens_after_repeated_failures(settings):
    cb = CircuitBreaker(get_recorder(settings.observability_db), settings)
    cb.record("prices:secondary", ok=False, run_id="a")
    cb.record("prices:secondary", ok=False, run_id="b")
    assert not cb.is_open("prices:secondary")  # below minimum events
    cb.record("prices:secondary", ok=True, run_id="c")
    assert cb.is_open("prices:secondary")  # 2/3 failed
    assert not cb.is_open("prices:primary")
