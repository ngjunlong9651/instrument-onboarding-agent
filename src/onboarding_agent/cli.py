"""Command line interface.

    agent run "Onboard the new UST US91282CJL54"      # pauses for approval if it wants to write
    agent pending                                      # runs waiting for a human
    agent approve <run_id> [--reject] [--comment ...]  # resume a paused run
    agent trace <run_id>                               # every node, LLM call, tool call and decision
    agent report                                       # completion rate, end reasons, p50/p95 per node, cost
    agent alerts                                       # exit code 1 if any alert fires
    agent demo                                         # run all scenarios end to end
    agent reset                                        # wipe local state
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from .config import PROJECT_ROOT, get_settings
from .observability import build_report, check_alerts, get_recorder

SCENARIOS = PROJECT_ROOT / "evals" / "scenarios.jsonl"


def _print_outcome(o) -> None:
    if o.status == "awaiting_approval":
        print(f"[{o.run_id}] waiting for approval")
        print(json.dumps(o.pending, indent=2, default=str))
        print(f"\nApprove with:  agent approve {o.run_id}\nReject with:   agent approve {o.run_id} --reject --comment '...'")
    else:
        print(f"[{o.run_id}] {o.end_reason}: {o.detail}")


def cmd_run(a) -> int:
    from .graph import Agent

    _print_outcome(Agent().run(a.request, auto_approve=a.auto_approve))
    return 0


def cmd_approve(a) -> int:
    from .graph import Agent

    _print_outcome(Agent().resume(a.run_id, approved=not a.reject, reviewer=a.reviewer, comment=a.comment))
    return 0


def cmd_pending(_) -> int:
    rows = get_recorder(get_settings().observability_db).awaiting_approval()
    if not rows:
        print("No runs waiting for approval.")
    for r in rows:
        print(f"{r['run_id']}  {r['workflow']:<22} {r['isin']}  {r['request']}")
    return 0


def cmd_trace(a) -> int:
    rec = get_recorder(get_settings().observability_db)
    run = rec.run(a.run_id)
    if not run:
        print(f"unknown run {a.run_id}")
        return 1
    print(json.dumps(run, indent=2))
    for sp in rec.spans_for(a.run_id):
        indent = "  " if sp["kind"] == "node" else "      "
        detail = {k: v for k, v in sp["detail"].items() if v not in (None, "", {}, [])}
        print(f"{indent}{sp['kind']:<8} {sp['name']:<26} {sp['status']:<22} {sp['latency_ms']:8.1f} ms  {json.dumps(detail, default=str)[:160]}")
    return 0


def cmd_report(a) -> int:
    s = get_settings()
    rep = build_report(get_recorder(s.observability_db), a.last)
    if rep.runs == 0:
        print("No finished runs yet. Try: agent demo")
        return 0
    print(f"Runs: {rep.runs}   task completion: {rep.completion_rate:.0%}   avg cost/run: ${rep.avg_cost_usd:.4f}   run p95: {rep.run_p95_ms:.0f} ms\n")
    print("End reasons:")
    for k, v in sorted(rep.end_reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24}{v:>4}")
    print("\nBy workflow:")
    for wf, d in rep.by_workflow.items():
        print(f"  {wf:<24}{d['runs']:>4} runs  completion {d['completion_rate']:.0%}  {d['end_reasons']}")
    print("\nLatency per node (ms):       p50       p95")
    for node in rep.node_p50_ms:
        print(f"  {node:<24}{rep.node_p50_ms[node]:>9.1f} {rep.node_p95_ms[node]:>9.1f}")
    return 0


def cmd_alerts(a) -> int:
    s = get_settings()
    alerts = check_alerts(build_report(get_recorder(s.observability_db), s.alert_window), s)
    for msg in alerts:
        print(f"ALERT: {msg}")
    if not alerts:
        print("No alerts.")
    return 1 if alerts else 0


def cmd_demo(_) -> int:
    from .graph import Agent

    # Start from a fresh booking system so onboarding scenarios aren't "already exists" on a second demo.
    # Observability history is kept, so `agent report` accumulates across demos.
    (get_settings().data_dir / "booking_system.json").unlink(missing_ok=True)
    agent = Agent()
    for line in SCENARIOS.read_text().splitlines():
        sc = json.loads(line)
        o = agent.run(sc["request"], auto_approve=True)
        mark = "ok " if o.end_reason == sc["expected"] else "MISMATCH"
        print(f"{mark} {o.end_reason:<22} {sc['request']}\n      -> {o.detail}")
    print()
    return cmd_report(argparse.Namespace(last=None))


def cmd_reset(_) -> int:
    s = get_settings()
    shutil.rmtree(s.data_dir, ignore_errors=True)
    print(f"Removed {s.data_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one request")
    r.add_argument("request")
    r.add_argument("--auto-approve", action="store_true", help="skip the human approval pause")
    r.set_defaults(fn=cmd_run)

    ap = sub.add_parser("approve", help="approve or reject a paused run")
    ap.add_argument("run_id")
    ap.add_argument("--reject", action="store_true")
    ap.add_argument("--reviewer", default="ops-analyst")
    ap.add_argument("--comment", default="")
    ap.set_defaults(fn=cmd_approve)

    sub.add_parser("pending", help="list runs waiting for approval").set_defaults(fn=cmd_pending)

    t = sub.add_parser("trace", help="show every span of a run")
    t.add_argument("run_id")
    t.set_defaults(fn=cmd_trace)

    rp = sub.add_parser("report", help="completion, end reasons, latency, cost")
    rp.add_argument("--last", type=int, default=None, help="only the last N runs")
    rp.set_defaults(fn=cmd_report)

    sub.add_parser("alerts", help="evaluate alert rules (exit 1 if any fire)").set_defaults(fn=cmd_alerts)
    sub.add_parser("demo", help="run every scenario with auto approval").set_defaults(fn=cmd_demo)
    sub.add_parser("reset", help="delete local state").set_defaults(fn=cmd_reset)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
