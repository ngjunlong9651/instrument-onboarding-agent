r"""The agent graph.

    retriever -> planner -> executor -> validator -> approval -> commit
         \__________\__________\___________\___________\________\__> finalize

Every node can end the run by setting `end_reason`; the router then sends it straight to `finalize`.
Read tools run in the executor. Write tools only run in `commit`, after the validator has produced the
exact payloads and a human (or --auto-approve) has approved them. The approval step uses LangGraph's
interrupt plus a SQLite checkpointer, so a run can pause and be resumed later from another process.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.runnables.config import ContextThreadPoolExecutor
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Settings, get_settings
from .guards import CircuitBreaker, check_price
from .models import (
    WRITE_STEPS,
    EndReason,
    Plan,
    PriceComparison,
    PriceQuote,
    Step,
    StepType,
    Workflow,
)
from .observability import Recorder, get_recorder, observed_node
from .planner import Planner, make_planner
from .retrieval import HybridRetriever
from .tools import call_tool
from .vendors import Services


class AgentState(TypedDict, total=False):
    run_id: str
    request: str
    release: str
    auto_approve: bool
    retrieved: list[dict[str, Any]]
    plan: dict[str, Any] | None
    observations: dict[str, dict[str, Any]]
    pending_writes: list[dict[str, Any]]
    evidence: list[str]
    approval: dict[str, Any] | None
    write_refs: list[str]
    end_reason: str | None
    end_detail: str


_TOOL_FAILURE_TO_END = {
    "timeout": EndReason.TIMEOUT,
    "not_found": EndReason.ESCALATED,
    "data_error": EndReason.REJECTED_BY_VALIDATOR,
    "bad_args": EndReason.TOOL_ERROR,
    "rejected": EndReason.TOOL_ERROR,
}


def _end(reason: EndReason, detail: str, **extra: Any) -> dict[str, Any]:
    return {"end_reason": reason.value, "end_detail": detail, **extra}


def _obs_key(step_type: StepType, source: str | None) -> str:
    return f"{step_type.value}:{source}" if source else step_type.value


class Nodes:
    def __init__(self, svc: Services, recorder: Recorder, retriever: HybridRetriever, planner: Planner, s: Settings):
        self.svc = svc
        self.recorder = recorder
        self.retriever = retriever
        self.planner = planner
        self.s = s
        self.breaker = CircuitBreaker(recorder, s)

    def _tool(self, state: AgentState, step: StepType, args: dict[str, Any]):
        return call_tool(step, args, self.svc, self.recorder, state["run_id"], self.s.tool_max_retries, self.s.tool_backoff_s)

    @observed_node("retriever")
    def retriever_node(self, state: AgentState) -> dict[str, Any]:
        hits = self.retriever.search(state["request"], k=3)
        return {"retrieved": [{"id": p.id, "score": round(score, 4)} for p, score in hits]}

    @observed_node("planner")
    def planner_node(self, state: AgentState) -> dict[str, Any]:
        candidates = [self.retriever.get(h["id"]) for h in state["retrieved"]]
        out = self.planner.plan(state["request"], candidates)
        self.recorder.add_usage(state["run_id"], out.usage.tokens_in, out.usage.tokens_out, out.usage.cost_usd)
        self.recorder.span(
            state["run_id"], "plan", "llm", "ok" if out.plan else "invalid", 0.0,
            {"model": out.model, "attempts": out.attempts, "error": out.error,
             "tokens_in": out.usage.tokens_in, "tokens_out": out.usage.tokens_out},
        )
        if out.plan is None:
            return _end(EndReason.ESCALATED, f"could not build a valid plan: {out.error}", plan=None)
        self.recorder.set_run_fields(state["run_id"], workflow=out.plan.workflow.value, isin=out.plan.isin)
        return {"plan": out.plan.model_dump(mode="json")}

    @observed_node("executor")
    def executor_node(self, state: AgentState) -> dict[str, Any]:
        plan = Plan.model_validate(state["plan"])
        obs: dict[str, dict[str, Any]] = {}
        batch: list[tuple[Step, dict[str, Any]]] = []

        def flush() -> dict[str, Any] | None:
            # Reads in a batch don't depend on each other, so they can run concurrently.
            # Results are still checked in plan order so the first failure wins deterministically.
            if not batch:
                return None
            if self.s.parallel_tools and len(batch) > 1:
                with ContextThreadPoolExecutor(max_workers=len(batch)) as pool:
                    results = list(pool.map(lambda sa: self._tool(state, sa[0].type, sa[1]), batch))
            else:
                results = [self._tool(state, st.type, a) for st, a in batch]
            for (st, _), res in zip(batch, results):
                if not res.ok:
                    reason = _TOOL_FAILURE_TO_END[res.error_kind]
                    return _end(reason, f"{st.type.value} failed ({res.error_kind}, {res.attempts} attempts): {res.error}", observations=obs)
                obs[_obs_key(st.type, st.source)] = res.data
            batch.clear()
            return None

        for step in plan.steps:
            if step.type in WRITE_STEPS:
                continue  # payloads are built by the validator and executed after approval
            if step.type == StepType.ESCALATE:
                if (ended := flush()):
                    return ended
                why = step.reason or f"procedure {plan.procedure_id} requires human handling"
                return _end(EndReason.ESCALATED, why, observations=obs)
            if step.type == StepType.COMPARE_SOURCES:
                if (ended := flush()):
                    return ended
                p, q = obs.get("get_vendor_prices:primary"), obs.get("get_vendor_prices:secondary")
                if not (p and q):
                    return _end(EndReason.TOOL_ERROR, "compare_price_sources needs both price sources", observations=obs)
                batch.append((step, {"primary": p, "secondary": q, "tolerance_pct": self.s.price_tolerance_pct}))
                if (ended := flush()):
                    return ended
                continue
            args = {"isin": plan.isin, "source": step.source} if step.type == StepType.GET_PRICES else {"isin": plan.isin}
            batch.append((step, args))
        if (ended := flush()):
            return ended
        return {"observations": obs}

    @observed_node("validator")
    def validator_node(self, state: AgentState) -> dict[str, Any]:
        plan = Plan.model_validate(state["plan"])
        obs = state.get("observations", {})
        planned = {s.type for s in plan.steps}
        evidence: list[str] = []
        writes: list[dict[str, Any]] = []

        if plan.workflow == Workflow.ONBOARDING:
            if obs.get("check_booking_system", {}).get("exists"):
                return _end(EndReason.ESCALATED, f"{plan.isin} already exists in the booking system; amendments go to static data")
            record = obs["lookup_reference_data"]
            evidence.append(f"reference data validated: {record['name']} {record['currency']} maturing {record['maturity_date']}")
            if StepType.CREATE_INSTRUMENT in planned:
                writes.append({"step": StepType.CREATE_INSTRUMENT.value, "args": record})
            return {"pending_writes": writes, "evidence": evidence}

        # Price workflows: check every quote before trusting any comparison made from it.
        quotes = {src: PriceQuote.model_validate(obs[f"get_vendor_prices:{src}"]) for src in ("primary", "secondary")
                  if f"get_vendor_prices:{src}" in obs}
        implausible, stale = [], []
        for src, q in quotes.items():
            feed = f"prices:{src}"
            g = check_price(q, self.s)
            self.breaker.record(feed, ok=not g.implausible, run_id=state["run_id"])
            implausible += g.implausible
            stale += g.stale
            if self.breaker.is_open(feed):
                return _end(EndReason.ESCALATED, f"circuit breaker open for {feed}: too many recent failures, writes stopped. {g.problems}")
        if implausible:
            return _end(EndReason.REJECTED_BY_VALIDATOR, "; ".join(implausible))
        if stale:
            return _end(EndReason.ESCALATED, "; ".join(stale))

        cmp = PriceComparison.model_validate(obs["compare_price_sources"])
        evidence.append(f"primary {cmp.primary} vs secondary {cmp.secondary}: diff {cmp.diff_pct:.3f}% (tolerance {self.s.price_tolerance_pct}%)")

        if plan.workflow == Workflow.PRICE_VALIDATION:
            if not cmp.within_tolerance:
                return _end(EndReason.ESCALATED, f"sources disagree by {cmp.diff_pct:.2f}%; not publishing")
            if StepType.PUBLISH_PRICE in planned:
                writes.append({"step": StepType.PUBLISH_PRICE.value, "args": {"quote": quotes["primary"].model_dump(mode="json")}})
        else:  # reconciliation
            if cmp.within_tolerance:
                return _end(EndReason.COMPLETED, "feeds agree within tolerance; no break", evidence=evidence)
            if StepType.FLAG_BREAK in planned:
                writes.append({"step": StepType.FLAG_BREAK.value, "args": {"comparison": cmp.model_dump()}})
        return {"pending_writes": writes, "evidence": evidence}

    @observed_node("approval")
    def approval_node(self, state: AgentState) -> dict[str, Any]:
        writes = state.get("pending_writes") or []
        if not writes:
            return _end(EndReason.COMPLETED, "nothing to write")
        if state.get("auto_approve"):
            decision = {"approved": True, "reviewer": "auto-approve", "comment": ""}
        else:
            self.recorder.set_run_fields(state["run_id"], status="awaiting_approval")
            decision = interrupt({"pending_writes": writes, "evidence": state.get("evidence", [])})
            self.recorder.set_run_fields(state["run_id"], status="running")
        self.recorder.span(state["run_id"], "human_approval", "approval", "approved" if decision.get("approved") else "rejected", 0.0, decision)
        if not decision.get("approved"):
            return _end(EndReason.REJECTED_BY_HUMAN, f"rejected by {decision.get('reviewer', 'reviewer')}: {decision.get('comment', '')}".strip(), approval=decision)
        return {"approval": decision}

    @observed_node("commit")
    def commit_node(self, state: AgentState) -> dict[str, Any]:
        refs = []
        for w in state["pending_writes"]:
            res = self._tool(state, StepType(w["step"]), w["args"])
            if not res.ok:
                return _end(EndReason.TOOL_ERROR, f"{w['step']} failed: {res.error}", write_refs=refs)
            refs.append(res.data["reference"])
        return _end(EndReason.COMPLETED, "written: " + ", ".join(refs), write_refs=refs)

    def finalize_node(self, state: AgentState) -> dict[str, Any]:
        self.recorder.end_run(state["run_id"], state["end_reason"], state.get("end_detail", ""))
        return {}


def _route(next_node: str):
    def router(state: AgentState) -> str:
        return "finalize" if state.get("end_reason") else next_node

    return router


def build_graph(nodes: Nodes, checkpointer=None):
    g = StateGraph(AgentState)
    order = [
        ("retriever", nodes.retriever_node),
        ("planner", nodes.planner_node),
        ("executor", nodes.executor_node),
        ("validator", nodes.validator_node),
        ("approval", nodes.approval_node),
        ("commit", nodes.commit_node),
    ]
    for name, fn in order:
        g.add_node(name, fn)
    g.add_node("finalize", nodes.finalize_node)
    g.add_edge(START, "retriever")
    for (name, _), (nxt, _) in zip(order, order[1:]):
        g.add_conditional_edges(name, _route(nxt), {nxt: nxt, "finalize": "finalize"})
    g.add_edge("commit", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


@dataclass
class RunOutcome:
    run_id: str
    status: str  # "ended" or "awaiting_approval"
    end_reason: str | None
    detail: str
    pending: dict[str, Any] | None = None


class Agent:
    def __init__(self, settings: Settings | None = None, services: Services | None = None, planner: Planner | None = None):
        self.s = settings or get_settings()
        self.recorder = get_recorder(self.s.observability_db)
        self.services = services or Services(self.s.data_dir)
        self.retriever = HybridRetriever()
        nodes = Nodes(self.services, self.recorder, self.retriever, planner or make_planner(self.s), self.s)
        conn = sqlite3.connect(self.s.checkpoint_db, check_same_thread=False)
        self.graph = build_graph(nodes, SqliteSaver(conn))

    def _config(self, run_id: str) -> dict[str, Any]:
        # Tags and metadata show up on the LangSmith trace when tracing is enabled.
        return {
            "configurable": {"thread_id": run_id},
            "run_name": "onboarding-agent",
            "tags": [f"release:{self.s.release}"],
            "metadata": {"run_id": run_id, "release": self.s.release},
        }

    def _outcome(self, run_id: str) -> RunOutcome:
        snap = self.graph.get_state(self._config(run_id))
        if snap.next:
            pending = None
            for task in snap.tasks:
                if task.interrupts:
                    pending = task.interrupts[0].value
            return RunOutcome(run_id, "awaiting_approval", None, "waiting for human approval", pending)
        v = snap.values
        return RunOutcome(run_id, "ended", v.get("end_reason"), v.get("end_detail", ""))

    def run(self, request: str, auto_approve: bool = False, run_id: str | None = None) -> RunOutcome:
        run_id = run_id or uuid.uuid4().hex[:12]
        self.recorder.start_run(run_id, request, self.s.release)
        state: AgentState = {"run_id": run_id, "request": request, "release": self.s.release, "auto_approve": auto_approve}
        self.graph.invoke(state, self._config(run_id))
        return self._outcome(run_id)

    def resume(self, run_id: str, approved: bool, reviewer: str = "reviewer", comment: str = "") -> RunOutcome:
        decision = {"approved": approved, "reviewer": reviewer, "comment": comment}
        self.graph.invoke(Command(resume=decision), self._config(run_id))
        return self._outcome(run_id)
