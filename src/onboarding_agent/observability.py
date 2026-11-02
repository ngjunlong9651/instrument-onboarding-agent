"""Run level observability, stored in SQLite so it works with no external service.

What every run records:
- one row in `runs`: workflow, ISIN, release, end reason, tokens and cost;
- one row in `spans` per graph node, LLM call, tool call and approval decision, with latency and status;
- feed events used by the circuit breaker.

When LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set, LangGraph also sends full traces to LangSmith,
tagged with the same run id, workflow and release (see graph.run_config).
"""

from __future__ import annotations

import functools
import json
import sqlite3
import statistics
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    request TEXT,
    workflow TEXT,
    isin TEXT,
    release TEXT,
    status TEXT,
    end_reason TEXT,
    end_detail TEXT,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    started_at TEXT,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    name TEXT,
    kind TEXT,
    status TEXT,
    latency_ms REAL,
    detail TEXT,
    started_at TEXT
);
CREATE TABLE IF NOT EXISTS feed_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed TEXT,
    ok INTEGER,
    run_id TEXT,
    at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur.fetchall()

    # Runs

    def start_run(self, run_id: str, request: str, release: str) -> None:
        self._exec(
            "INSERT OR IGNORE INTO runs (run_id, request, release, status, started_at) VALUES (?, ?, ?, 'running', ?)",
            (run_id, request, release, _now()),
        )

    def set_run_fields(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE runs SET {cols} WHERE run_id = ?", (*fields.values(), run_id))

    def add_usage(self, run_id: str, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self._exec(
            "UPDATE runs SET tokens_in = tokens_in + ?, tokens_out = tokens_out + ?, cost_usd = cost_usd + ? WHERE run_id = ?",
            (tokens_in, tokens_out, cost_usd, run_id),
        )

    def end_run(self, run_id: str, end_reason: str, detail: str = "") -> None:
        self.set_run_fields(run_id, status="ended", end_reason=end_reason, end_detail=detail, ended_at=_now())

    # Spans

    def span(self, run_id: str, name: str, kind: str, status: str, latency_ms: float, detail: dict | None = None) -> None:
        self._exec(
            "INSERT INTO spans (run_id, name, kind, status, latency_ms, detail, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, name, kind, status, latency_ms, json.dumps(detail or {}, default=str), _now()),
        )

    def spans_for(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT name, kind, status, latency_ms, detail FROM spans WHERE run_id = ? ORDER BY id", (run_id,)
        )
        return [dict(name=r[0], kind=r[1], status=r[2], latency_ms=r[3], detail=json.loads(r[4])) for r in rows]

    # Circuit breaker events

    def feed_event(self, feed: str, ok: bool, run_id: str) -> None:
        self._exec("INSERT INTO feed_events (feed, ok, run_id, at) VALUES (?, ?, ?, ?)", (feed, int(ok), run_id, _now()))

    def recent_feed_events(self, feed: str, window: int) -> list[bool]:
        rows = self._exec("SELECT ok FROM feed_events WHERE feed = ? ORDER BY id DESC LIMIT ?", (feed, window))
        return [bool(r[0]) for r in rows]

    # Queries

    def run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._exec(
            "SELECT run_id, request, workflow, isin, release, status, end_reason, end_detail, tokens_in, tokens_out, cost_usd "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        )
        if not rows:
            return None
        keys = ["run_id", "request", "workflow", "isin", "release", "status", "end_reason", "end_detail", "tokens_in", "tokens_out", "cost_usd"]
        return dict(zip(keys, rows[0]))

    def awaiting_approval(self) -> list[dict[str, Any]]:
        rows = self._exec("SELECT run_id, request, workflow, isin FROM runs WHERE status = 'awaiting_approval' ORDER BY started_at")
        return [dict(run_id=r[0], request=r[1], workflow=r[2], isin=r[3]) for r in rows]

    def ended_runs(self, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT run_id, workflow, end_reason, cost_usd FROM runs WHERE status = 'ended' ORDER BY ended_at DESC"
        rows = self._exec(sql + (f" LIMIT {int(limit)}" if limit else ""))
        return [dict(run_id=r[0], workflow=r[1] or "unknown", end_reason=r[2], cost_usd=r[3] or 0.0) for r in rows]

    def node_latencies(self, run_ids: list[str]) -> dict[str, list[float]]:
        if not run_ids:
            return {}
        marks = ",".join("?" * len(run_ids))
        rows = self._exec(f"SELECT name, latency_ms FROM spans WHERE kind = 'node' AND run_id IN ({marks})", tuple(run_ids))
        out: dict[str, list[float]] = defaultdict(list)
        for name, ms in rows:
            out[name].append(ms)
        return out

    def run_latencies(self, run_ids: list[str]) -> list[float]:
        if not run_ids:
            return []
        marks = ",".join("?" * len(run_ids))
        rows = self._exec(
            f"SELECT run_id, SUM(latency_ms) FROM spans WHERE kind = 'node' AND run_id IN ({marks}) GROUP BY run_id",
            tuple(run_ids),
        )
        return [r[1] for r in rows]


_recorders: dict[Path, Recorder] = {}


def get_recorder(path: Path) -> Recorder:
    if path not in _recorders:
        _recorders[path] = Recorder(path)
    return _recorders[path]


def observed_node(name: str) -> Callable:
    """Decorator for graph node methods: records latency and status of every node execution."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, state: dict[str, Any]) -> dict[str, Any]:
            t0 = time.perf_counter()
            status = "ok"
            try:
                out = fn(self, state)
                if out.get("end_reason"):
                    status = out["end_reason"]
                return out
            except Exception as e:  # GraphInterrupt is re-raised untouched below
                status = "interrupted" if type(e).__name__ == "GraphInterrupt" else "error"
                raise
            finally:
                self.recorder.span(state["run_id"], name, "node", status, (time.perf_counter() - t0) * 1000)

        return wrapper

    return deco


# Reporting

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


@dataclass
class Report:
    runs: int
    completion_rate: float
    end_reasons: dict[str, int]
    by_workflow: dict[str, dict[str, Any]]
    node_p50_ms: dict[str, float]
    node_p95_ms: dict[str, float]
    run_p95_ms: float
    avg_cost_usd: float


def build_report(rec: Recorder, limit: int | None = None) -> Report:
    runs = rec.ended_runs(limit)
    ids = [r["run_id"] for r in runs]
    reasons = Counter(r["end_reason"] for r in runs)
    by_wf: dict[str, dict[str, Any]] = {}
    for wf in sorted({r["workflow"] for r in runs}):
        sub = [r for r in runs if r["workflow"] == wf]
        done = sum(r["end_reason"] == "completed" for r in sub)
        by_wf[wf] = {"runs": len(sub), "completion_rate": done / len(sub), "end_reasons": dict(Counter(r["end_reason"] for r in sub))}
    lat = rec.node_latencies(ids)
    return Report(
        runs=len(runs),
        completion_rate=(reasons["completed"] / len(runs)) if runs else 0.0,
        end_reasons=dict(reasons),
        by_workflow=by_wf,
        node_p50_ms={k: percentile(v, 50) for k, v in sorted(lat.items())},
        node_p95_ms={k: percentile(v, 95) for k, v in sorted(lat.items())},
        run_p95_ms=percentile(rec.run_latencies(ids), 95),
        avg_cost_usd=(sum(r["cost_usd"] for r in runs) / len(runs)) if runs else 0.0,
    )


def check_alerts(rep: Report, settings) -> list[str]:
    alerts = []
    if rep.runs == 0:
        return alerts
    if rep.completion_rate < settings.alert_min_completion_rate:
        alerts.append(f"completion rate {rep.completion_rate:.0%} below {settings.alert_min_completion_rate:.0%}")
    tool_errors = (rep.end_reasons.get("tool_error", 0) + rep.end_reasons.get("timeout", 0)) / rep.runs
    if tool_errors > settings.alert_max_tool_error_rate:
        alerts.append(f"tool error + timeout rate {tool_errors:.0%} above {settings.alert_max_tool_error_rate:.0%}")
    if rep.run_p95_ms > settings.alert_max_p95_ms:
        alerts.append(f"run p95 latency {rep.run_p95_ms:.0f} ms above {settings.alert_max_p95_ms:.0f} ms")
    return alerts
