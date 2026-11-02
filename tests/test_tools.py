from onboarding_agent.models import StepType
from onboarding_agent.observability import get_recorder
from onboarding_agent.tools import call_tool
from onboarding_agent.vendors import Services


def _call(settings, step, args, svc=None):
    svc = svc or Services(settings.data_dir)
    rec = get_recorder(settings.observability_db)
    return call_tool(step, args, svc, rec, "t1", max_retries=2, backoff_s=0)


def test_bad_args_return_readable_error_without_calling_vendor(settings):
    res = _call(settings, StepType.GET_PRICES, {"isin": "not-an-isin", "source": "tertiary"})
    assert not res.ok and res.error_kind == "bad_args" and res.attempts == 0
    assert "isin" in res.error and "source" in res.error


def test_transient_timeout_is_retried(settings):
    res = _call(settings, StepType.LOOKUP_REFERENCE, {"isin": "DE0001102580"})
    assert res.ok and res.attempts == 2


def test_permanent_timeout_gives_up_after_retries(settings):
    res = _call(settings, StepType.LOOKUP_REFERENCE, {"isin": "FR0013508470"})
    assert not res.ok and res.error_kind == "timeout" and res.attempts == 3


def test_bad_vendor_data_is_not_retried(settings):
    res = _call(settings, StepType.LOOKUP_REFERENCE, {"isin": "SG3260987654"})
    assert not res.ok and res.error_kind == "data_error" and res.attempts == 1
    assert "maturity_date" in res.error


def test_not_found(settings):
    res = _call(settings, StepType.LOOKUP_REFERENCE, {"isin": "XS1111111110"})
    assert res.error_kind == "not_found"


def test_tool_calls_are_recorded(settings):
    _call(settings, StepType.CHECK_BOOKING, {"isin": "US91282CJL54"})
    spans = get_recorder(settings.observability_db).spans_for("t1")
    assert spans[-1]["name"] == "check_booking_system" and spans[-1]["detail"]["args"] == {"isin": "US91282CJL54"}
