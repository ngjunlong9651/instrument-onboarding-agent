"""Release gate.

1. Retrieval: Recall@1, Recall@3 and MRR on the golden set, for BM25, vector and hybrid.
2. Scenarios: run every scenario end to end (offline planner, auto approval) and check the end reason.

Fails (exit 1) if hybrid retrieval drops below baseline.json by more than the tolerance, or any scenario
ends differently than expected. Run with --update-baseline after an intentional improvement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline.json"
TOLERANCE = 0.02


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def retrieval_metrics(mode: str) -> dict[str, float]:
    from onboarding_agent.retrieval import HybridRetriever

    retriever = HybridRetriever()
    golden = load_jsonl(HERE / "golden_set.jsonl")
    r1 = r3 = rr = 0.0
    misses = []
    for g in golden:
        ranked = [p.id for p, _ in retriever.rank(g["query"], mode)]
        pos = ranked.index(g["expected"]) + 1
        r1 += pos == 1
        r3 += pos <= 3
        rr += 1 / pos
        if pos > 1:
            misses.append((g["query"], g["expected"], ranked[0]))
    n = len(golden)
    return {"recall@1": round(r1 / n, 4), "recall@3": round(r3 / n, 4), "mrr": round(rr / n, 4), "_misses": misses}


def scenario_results() -> list[tuple[dict, str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["AGENT_DATA_DIR"] = tmp
        os.environ["AGENT_OFFLINE"] = "1"
        from onboarding_agent.graph import Agent

        agent = Agent()
        out = []
        for sc in load_jsonl(HERE / "scenarios.jsonl"):
            o = agent.run(sc["request"], auto_approve=True)
            out.append((sc, o.end_reason, o.detail))
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    print("Retrieval on golden set")
    print(f"  {'mode':<8}{'recall@1':>10}{'recall@3':>10}{'mrr':>8}")
    results = {}
    for mode in ("bm25", "vector", "hybrid"):
        m = retrieval_metrics(mode)
        results[mode] = {k: v for k, v in m.items() if not k.startswith("_")}
        print(f"  {mode:<8}{m['recall@1']:>10.3f}{m['recall@3']:>10.3f}{m['mrr']:>8.3f}")
        if args.show_misses:
            for q, exp, got in m["_misses"]:
                print(f"      miss: {q!r} expected {exp}, top {got}")

    print("\nScenarios (offline planner, auto approval)")
    scen = scenario_results()
    passed = 0
    for sc, got, detail in scen:
        ok = got == sc["expected"]
        passed += ok
        print(f"  {'pass' if ok else 'FAIL'}  {sc['expected']:<22} {sc['request']}" + ("" if ok else f"\n        got {got}: {detail}"))
    completion = sum(got == "completed" for _, got, _ in scen) / len(scen)
    print(f"  {passed}/{len(scen)} scenarios behaved as expected; task completion {completion:.0%}")

    if args.update_baseline:
        BASELINE.write_text(json.dumps(results["hybrid"], indent=2) + "\n")
        print(f"\nBaseline updated: {results['hybrid']}")
        return 0

    failures = []
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        for metric in ("recall@3", "mrr"):
            if results["hybrid"][metric] < base[metric] - TOLERANCE:
                failures.append(f"hybrid {metric} {results['hybrid'][metric]:.3f} < baseline {base[metric]:.3f} - {TOLERANCE}")
    else:
        failures.append("no baseline.json; run with --update-baseline")
    if passed != len(scen):
        failures.append(f"{len(scen) - passed} scenario(s) ended differently than expected")

    print("\nGATE:", "PASS" if not failures else "FAIL")
    for f in failures:
        print("  -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
