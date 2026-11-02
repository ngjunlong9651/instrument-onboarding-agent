"""Typed tools.

Each tool has a Pydantic args model and a Pydantic result model. `call_tool` is the only way tools run:
- bad arguments come back as a readable validation error (so an LLM caller can correct itself);
- transient vendor failures are retried with backoff;
- data errors are never retried, because retrying bad data returns the same bad data;
- the vendor response is validated before anyone sees it.
Write tools are flagged so the graph only runs them after validation and human approval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError

from .models import (
    ISIN_PATTERN,
    BookingStatus,
    InstrumentRecord,
    PriceComparison,
    PriceQuote,
    PriceSource,
    StepType,
    WriteResult,
)
from .observability import Recorder
from .vendors import NotFoundError, Services, VendorError, VendorTimeout, adapt_price, adapt_security


class IsinArgs(BaseModel):
    isin: str = Field(pattern=ISIN_PATTERN)


class PriceArgs(BaseModel):
    isin: str = Field(pattern=ISIN_PATTERN)
    source: PriceSource


class CompareArgs(BaseModel):
    primary: PriceQuote
    secondary: PriceQuote
    tolerance_pct: float = Field(gt=0)


class PublishArgs(BaseModel):
    quote: PriceQuote


class BreakArgs(BaseModel):
    comparison: PriceComparison


@dataclass(frozen=True)
class ToolSpec:
    name: StepType
    args_model: type[BaseModel]
    result_model: type[BaseModel]
    fn: Callable[[Services, Any], Any]
    writes: bool = False


def _lookup(svc: Services, a: IsinArgs) -> InstrumentRecord:
    return adapt_security(svc.reference.get_security(a.isin))


def _check_booking(svc: Services, a: IsinArgs) -> BookingStatus:
    return BookingStatus(isin=a.isin, exists=svc.booking.exists(a.isin))


def _get_prices(svc: Services, a: PriceArgs) -> PriceQuote:
    return adapt_price(svc.pricing.get_eod_price(a.isin, a.source))


def _compare(_: Services, a: CompareArgs) -> PriceComparison:
    diff = abs(a.primary.clean_price - a.secondary.clean_price) / a.primary.clean_price * 100
    return PriceComparison(
        isin=a.primary.isin,
        primary=a.primary.clean_price,
        secondary=a.secondary.clean_price,
        diff_pct=round(diff, 4),
        within_tolerance=diff <= a.tolerance_pct,
    )


def _create(svc: Services, a: InstrumentRecord) -> WriteResult:
    return WriteResult(ok=True, reference=svc.booking.create_instrument(a))


def _publish(svc: Services, a: PublishArgs) -> WriteResult:
    return WriteResult(ok=True, reference=svc.booking.publish_price(a.quote.isin, a.quote.clean_price, a.quote.as_of.isoformat()))


def _flag(svc: Services, a: BreakArgs) -> WriteResult:
    return WriteResult(ok=True, reference=svc.booking.flag_break(a.comparison.isin, a.comparison.model_dump()))


TOOLS: dict[StepType, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec(StepType.LOOKUP_REFERENCE, IsinArgs, InstrumentRecord, _lookup),
        ToolSpec(StepType.CHECK_BOOKING, IsinArgs, BookingStatus, _check_booking),
        ToolSpec(StepType.GET_PRICES, PriceArgs, PriceQuote, _get_prices),
        ToolSpec(StepType.COMPARE_SOURCES, CompareArgs, PriceComparison, _compare),
        ToolSpec(StepType.CREATE_INSTRUMENT, InstrumentRecord, WriteResult, _create, writes=True),
        ToolSpec(StepType.PUBLISH_PRICE, PublishArgs, WriteResult, _publish, writes=True),
        ToolSpec(StepType.FLAG_BREAK, BreakArgs, WriteResult, _flag, writes=True),
    ]
}


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    # bad_args: caller's fault, fixable by the caller. data_error: vendor returned something wrong.
    # not_found: nothing to work with. timeout: retries exhausted. rejected: the write was refused.
    error_kind: str | None = None
    attempts: int = 0


def format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


@traceable(run_type="tool", name="call_tool")
def call_tool(
    step: StepType,
    raw_args: dict[str, Any],
    svc: Services,
    recorder: Recorder,
    run_id: str,
    max_retries: int = 2,
    backoff_s: float = 0.05,
) -> ToolResult:
    spec = TOOLS[step]
    t0 = time.perf_counter()

    def done(res: ToolResult) -> ToolResult:
        recorder.span(
            run_id,
            step.value,
            "tool",
            "ok" if res.ok else res.error_kind or "error",
            (time.perf_counter() - t0) * 1000,
            {"args": raw_args, "attempts": res.attempts, "error": res.error, "writes": spec.writes},
        )
        return res

    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as e:
        return done(ToolResult(tool=step.value, ok=False, error=format_validation_error(e), error_kind="bad_args"))

    attempts = 0
    while True:
        attempts += 1
        try:
            out = spec.fn(svc, args)
            break
        except VendorTimeout as e:
            if attempts > max_retries:
                return done(ToolResult(tool=step.value, ok=False, error=str(e), error_kind="timeout", attempts=attempts))
            time.sleep(backoff_s * 2 ** (attempts - 1))
        except NotFoundError as e:
            return done(ToolResult(tool=step.value, ok=False, error=str(e), error_kind="not_found", attempts=attempts))
        except VendorError as e:
            return done(ToolResult(tool=step.value, ok=False, error=str(e), error_kind="rejected", attempts=attempts))
        except ValidationError as e:
            # The adapter could not build a valid record from the vendor payload.
            return done(ToolResult(tool=step.value, ok=False, error=format_validation_error(e), error_kind="data_error", attempts=attempts))

    try:
        result = spec.result_model.model_validate(out.model_dump() if isinstance(out, BaseModel) else out)
    except ValidationError as e:
        return done(ToolResult(tool=step.value, ok=False, error=format_validation_error(e), error_kind="data_error", attempts=attempts))
    return done(ToolResult(tool=step.value, ok=True, data=result.model_dump(mode="json"), attempts=attempts))
