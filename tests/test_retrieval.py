import json
from pathlib import Path

from onboarding_agent.retrieval import HybridRetriever

GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden_set.jsonl"


def test_hybrid_recall_at_3_on_golden_set():
    r = HybridRetriever()
    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines()]
    hits = sum(g["expected"] in [p.id for p, _ in r.search(g["query"], k=3)] for g in rows)
    assert hits / len(rows) >= 0.95


def test_every_procedure_parses_with_known_steps():
    from onboarding_agent.planner import procedure_steps

    for p in HybridRetriever().procedures:
        assert procedure_steps(p)
