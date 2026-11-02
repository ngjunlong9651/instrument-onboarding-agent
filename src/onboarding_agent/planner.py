"""Planners turn a request plus retrieved procedures into a typed Plan.

ClaudePlanner asks the model for a Plan via structured output. If the output fails validation, the validation
error goes back to the model once so it can correct the exact field, instead of guessing again from scratch.

ScriptedPlanner is deterministic and needs no API key. It takes the ISIN from the request and the steps from the
top ranked procedure. It is what the tests and the CI eval use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from .config import Settings
from .models import Plan, Step, StepType
from .retrieval import Procedure
from .tools import format_validation_error

ISIN_IN_TEXT = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b")


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class PlanOutcome:
    plan: Plan | None
    error: str | None = None
    attempts: int = 1
    usage: Usage = field(default_factory=Usage)
    model: str = "scripted"


class Planner(Protocol):
    def plan(self, request: str, candidates: list[Procedure]) -> PlanOutcome: ...


def procedure_steps(proc: Procedure) -> list[Step]:
    steps = []
    for raw in proc.steps:
        name, _, source = raw.partition(":")
        steps.append(Step(type=StepType(name), source=source or None))
    return steps


class ScriptedPlanner:
    def plan(self, request: str, candidates: list[Procedure]) -> PlanOutcome:
        isins = ISIN_IN_TEXT.findall(request)
        if not isins:
            return PlanOutcome(plan=None, error="no valid ISIN found in the request")
        if len(set(isins)) > 1:
            return PlanOutcome(plan=None, error=f"request mentions several ISINs {sorted(set(isins))}; one per run")
        proc = candidates[0]
        try:
            plan = Plan(workflow=proc.workflow, isin=isins[0], procedure_id=proc.id, steps=procedure_steps(proc))
        except ValidationError as e:
            return PlanOutcome(plan=None, error=format_validation_error(e))
        return PlanOutcome(plan=plan)


SYSTEM_PROMPT = """You plan work for a market data operations agent.

Pick the ONE procedure below that matches the request and return a plan:
- workflow and procedure_id must come from the chosen procedure;
- isin is the 12 character ISIN in the request (do not invent one);
- steps must follow the chosen procedure's steps, in order, using only these step types:
  {step_types}
- get_vendor_prices needs source "primary" or "secondary";
- if nothing fits, or the request is ambiguous, use the closest procedure with a single escalate_to_human step and a reason.

Procedures:
{procedures}"""


class ClaudePlanner:
    def __init__(self, settings: Settings):
        from langchain_anthropic import ChatAnthropic  # optional dependency: pip install -e ".[llm]"

        self.s = settings
        llm = ChatAnthropic(model=settings.model, temperature=0, max_tokens=1024)
        self.llm = llm.with_structured_output(Plan, include_raw=True)

    def _cost(self, tin: int, tout: int) -> float:
        return tin / 1e6 * self.s.cost_per_mtok_in + tout / 1e6 * self.s.cost_per_mtok_out

    def plan(self, request: str, candidates: list[Procedure]) -> PlanOutcome:
        procs = "\n\n".join(
            json.dumps({"id": p.id, "title": p.title, "workflow": p.workflow, "steps": p.steps, "description": p.body})
            for p in candidates
        )
        system = SYSTEM_PROMPT.format(step_types=", ".join(s.value for s in StepType), procedures=procs)
        messages: list = [("system", system), ("human", request)]
        usage = Usage()
        known = {p.id for p in candidates}
        error = None
        for attempt in range(1, self.s.planner_max_attempts + 1):
            out = self.llm.invoke(messages)
            meta = getattr(out["raw"], "usage_metadata", None) or {}
            usage.tokens_in += meta.get("input_tokens", 0)
            usage.tokens_out += meta.get("output_tokens", 0)
            usage.cost_usd = self._cost(usage.tokens_in, usage.tokens_out)
            plan: Plan | None = out.get("parsed")
            perr = out.get("parsing_error")
            if perr is not None:
                error = format_validation_error(perr) if isinstance(perr, ValidationError) else str(perr)
            elif plan is not None and plan.procedure_id not in known:
                error = f"procedure_id {plan.procedure_id!r} is not one of {sorted(known)}"
            elif plan is not None and plan.isin not in request:
                error = f"isin {plan.isin} does not appear in the request"
            elif plan is not None:
                return PlanOutcome(plan=plan, attempts=attempt, usage=usage, model=self.s.model)
            else:
                error = "model returned no plan"
            calls = getattr(out["raw"], "tool_calls", None) or []
            previous = json.dumps(calls[0]["args"], default=str) if calls else "(no plan)"
            # Feed the exact validation error back so the model fixes that field rather than re-planning blind.
            messages.append(
                ("human", f"Your previous plan was {previous}. It failed validation: {error}. "
                          "Fix only that and return the corrected plan.")
            )
        return PlanOutcome(plan=None, error=error, attempts=self.s.planner_max_attempts, usage=usage, model=self.s.model)


def make_planner(settings: Settings) -> Planner:
    return ClaudePlanner(settings) if settings.use_llm else ScriptedPlanner()
