"""Hybrid retrieval over operating procedures: BM25 (keyword) + vector similarity, fused with reciprocal rank fusion.

The vector leg uses character n-gram TF-IDF by default so the project runs offline with no model download.
Set AGENT_EMBEDDINGS=sentence-transformers (and install it) to use a neural embedding model instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

PROCEDURE_DIR = Path(__file__).resolve().parent / "procedures"

Mode = Literal["bm25", "vector", "hybrid"]

_STOP = set("a an the of for to in on and or is are be with this that it by as at from".split())


@dataclass
class Procedure:
    id: str
    title: str
    workflow: str
    steps: list[str]
    body: str
    path: Path = field(repr=False)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"


def _parse(path: Path) -> Procedure:
    raw = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        raise ValueError(f"{path} is missing front matter")
    meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
    return Procedure(
        id=meta["id"],
        title=meta["title"],
        workflow=meta["workflow"],
        steps=[s.strip() for s in meta["steps"].split(",")],
        body=m.group(2).strip(),
        path=path,
    )


def load_procedures(directory: Path = PROCEDURE_DIR) -> list[Procedure]:
    return [_parse(p) for p in sorted(directory.glob("*.md"))]


def _stem(t: str) -> str:
    """Light suffix stripping so 'Bunds'/'Bund' and 'onboarding'/'onboard' match. Good enough for short procedure docs."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class _TfidfEmbedder:
    def __init__(self, corpus: list[str]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return (self.matrix @ q.T).toarray().ravel()


class _SentenceTransformerEmbedder:
    def __init__(self, corpus: list[str]):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(os.getenv("AGENT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        self.matrix = self.model.encode(corpus, normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.model.encode([query], normalize_embeddings=True)[0]


class HybridRetriever:
    def __init__(self, procedures: list[Procedure] | None = None, rrf_k: int = 60):
        self.procedures = procedures or load_procedures()
        corpus = [p.text for p in self.procedures]
        self.bm25 = BM25Okapi([tokenize(t) for t in corpus])
        use_st = os.getenv("AGENT_EMBEDDINGS") == "sentence-transformers"
        self.embedder = _SentenceTransformerEmbedder(corpus) if use_st else _TfidfEmbedder(corpus)
        self.rrf_k = rrf_k

    def _ranking(self, scores: np.ndarray) -> list[int]:
        return list(np.argsort(-scores, kind="stable"))

    def rank(self, query: str, mode: Mode = "hybrid") -> list[tuple[Procedure, float]]:
        bm = np.asarray(self.bm25.get_scores(tokenize(query)))
        vec = self.embedder.scores(query)
        if mode == "bm25":
            order, scores = self._ranking(bm), bm
        elif mode == "vector":
            order, scores = self._ranking(vec), vec
        else:
            fused = np.zeros(len(self.procedures))
            for ranking in (self._ranking(bm), self._ranking(vec)):
                for pos, idx in enumerate(ranking):
                    fused[idx] += 1.0 / (self.rrf_k + pos + 1)
            order, scores = self._ranking(fused), fused
        return [(self.procedures[i], float(scores[i])) for i in order]

    def search(self, query: str, k: int = 3, mode: Mode = "hybrid") -> list[tuple[Procedure, float]]:
        return self.rank(query, mode)[:k]

    def get(self, procedure_id: str) -> Procedure | None:
        return next((p for p in self.procedures if p.id == procedure_id), None)
