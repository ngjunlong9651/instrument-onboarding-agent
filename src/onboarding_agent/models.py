"""Typed domain models. Everything that crosses a boundary (LLM output, tool args, vendor responses) is validated here."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ISIN_PATTERN = r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$"


class Workflow(str, Enum):
    ONBOARDING = "instrument_onboarding"
    PRICE_VALIDATION = "price_validation"
    RECONCILIATION = "vendor_reconciliation"


class StepType(str, Enum):
    """The planner may only use these steps. Anything else fails validation."""

    LOOKUP_REFERENCE = "lookup_reference_data"
    CHECK_BOOKING = "check_booking_system"
    GET_PRICES = "get_vendor_prices"
    COMPARE_SOURCES = "compare_price_sources"
    CREATE_INSTRUMENT = "create_instrument"
    PUBLISH_PRICE = "publish_validated_price"
    FLAG_BREAK = "flag_reconciliation_break"
    ESCALATE = "escalate_to_human"


WRITE_STEPS = {StepType.CREATE_INSTRUMENT, StepType.PUBLISH_PRICE, StepType.FLAG_BREAK}


class EndReason(str, Enum):
    COMPLETED = "completed"
    ESCALATED = "escalated"
    REJECTED_BY_VALIDATOR = "rejected_by_validator"
    REJECTED_BY_HUMAN = "rejected_by_human"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"


PriceSource = Literal["primary", "secondary"]


class Step(BaseModel):
    type: StepType
    source: PriceSource | None = Field(default=None, description="Only for get_vendor_prices.")
    reason: str | None = Field(default=None, description="Only for escalate_to_human.")

    @model_validator(mode="after")
    def _check_fields(self) -> "Step":
        if self.type == StepType.GET_PRICES and self.source is None:
            raise ValueError("get_vendor_prices needs a source: 'primary' or 'secondary'")
        return self


class Plan(BaseModel):
    workflow: Workflow
    isin: str = Field(pattern=ISIN_PATTERN, description="12 character ISIN taken from the request.")
    procedure_id: str
    steps: list[Step] = Field(min_length=1)


class InstrumentRecord(BaseModel):
    isin: str = Field(pattern=ISIN_PATTERN)
    name: str = Field(min_length=1)
    instrument_type: Literal["government_bond", "corporate_bond", "frn"]
    issuer: str = Field(min_length=1)
    currency: Literal["USD", "SGD", "EUR", "GBP", "AUD", "JPY"]
    coupon_rate: float = Field(ge=0, le=25)
    issue_date: date
    maturity_date: date
    day_count: Literal["ACT/ACT", "30/360", "ACT/360", "ACT/365"]

    @model_validator(mode="after")
    def _maturity_after_issue(self) -> "InstrumentRecord":
        if self.maturity_date <= self.issue_date:
            raise ValueError(f"maturity_date {self.maturity_date} is not after issue_date {self.issue_date}")
        return self


class PriceQuote(BaseModel):
    isin: str = Field(pattern=ISIN_PATTERN)
    source: PriceSource
    clean_price: float = Field(gt=0)
    as_of: datetime


class BookingStatus(BaseModel):
    isin: str
    exists: bool


class PriceComparison(BaseModel):
    isin: str
    primary: float
    secondary: float
    diff_pct: float
    within_tolerance: bool


class WriteResult(BaseModel):
    ok: bool
    reference: str
