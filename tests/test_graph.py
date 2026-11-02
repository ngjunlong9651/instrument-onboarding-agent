import json

from onboarding_agent.observability import build_report


def test_onboarding_pauses_for_approval_then_writes(agent):
    o = agent.run("Onboard the new US Treasury US91282CJL54 into the booking system")
    assert o.status == "awaiting_approval"
    assert o.pending["pending_writes"][0]["step"] == "create_instrument"
    assert not agent.services.booking.exists("US91282CJL54")  # nothing written before approval

    done = agent.resume(o.run_id, approved=True, reviewer="ops")
    assert done.end_reason == "completed"
    assert agent.services.booking.exists("US91282CJL54")


def test_human_rejection_writes_nothing(agent):
    o = agent.run("Run end of day price validation for XS2010031234")
    done = agent.resume(o.run_id, approved=False, reviewer="ops", comment="looks off")
    assert done.end_reason == "rejected_by_human"
    assert "XS2010031234" not in agent.services.booking.snapshot()["prices"]


def test_format_drift_does_not_create_false_breaks(agent):
    o = agent.run("Reconcile the primary and secondary vendor price feeds for XS3333333336", auto_approve=True)
    assert o.end_reason == "rejected_by_validator"
    assert agent.services.booking.snapshot()["breaks"] == []


def test_breaker_stops_writes_after_repeated_drift(agent):
    reasons = [agent.run("Validate today's EOD price for XS3333333336", auto_approve=True).end_reason for _ in range(4)]
    assert reasons[0] == "rejected_by_validator"
    assert reasons[-1] == "escalated"


def test_real_break_is_flagged(agent):
    o = agent.run("Reconcile the primary and secondary vendor price feeds for XS2222222228", auto_approve=True)
    assert o.end_reason == "completed"
    assert agent.services.booking.snapshot()["breaks"][0]["isin"] == "XS2222222228"


def test_every_run_has_end_reason_and_node_spans(agent):
    agent.run("Onboard the French OAT FR0013508470", auto_approve=True)
    agent.run("Onboard the new bond please", auto_approve=True)
    rep = build_report(agent.recorder)
    assert rep.runs == 2
    assert rep.end_reasons == {"timeout": 1, "escalated": 1}
    assert {"retriever", "planner"} <= set(rep.node_p95_ms)


def test_planner_validation_error_is_fed_back_and_corrected(settings):
    """A fake LLM returns an invalid plan first; the corrected second attempt is accepted."""
    from onboarding_agent.planner import ClaudePlanner

    class FakeRaw:
        def __init__(self, args):
            self.tool_calls = [{"args": args}]
            self.usage_metadata = {"input_tokens": 100, "output_tokens": 20}

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages[:])
            from onboarding_agent.models import Plan

            good = {"workflow": "instrument_onboarding", "isin": "US91282CJL54", "procedure_id": "onboard-government-bond",
                    "steps": [{"type": "lookup_reference_data"}]}
            if len(self.calls) == 1:
                bad = {**good, "isin": "US91282CJL5"}  # 11 chars
                from pydantic import ValidationError
                try:
                    Plan.model_validate(bad)
                except ValidationError as e:
                    return {"raw": FakeRaw(bad), "parsed": None, "parsing_error": e}
            return {"raw": FakeRaw(good), "parsed": Plan.model_validate(good), "parsing_error": None}

    from onboarding_agent.retrieval import HybridRetriever

    planner = ClaudePlanner.__new__(ClaudePlanner)
    planner.s = settings
    planner.llm = FakeLLM()
    cands = [p for p, _ in HybridRetriever().search("onboard treasury", k=3)]
    out = planner.plan("Onboard US91282CJL54", cands)
    assert out.plan is not None and out.attempts == 2
    assert "isin" in json.dumps(planner.llm.calls[1][-1])
    assert out.usage.tokens_in == 200
