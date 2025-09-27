"""Lightweight BM25 index for Lexicomp chunk retrieval.

Keeps an inverted index in memory and supports scoring queries against
chunk-level documents produced by the ingestion pipeline.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

TokenizeFn = Callable[[str], List[str]]

_DEFAULT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def default_tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase and keep alphanumeric tokens."""
    return [match.group(0).lower() for match in _DEFAULT_TOKEN_PATTERN.finditer(text)]


@dataclass
class Posting:
    doc_id: int
    term_freq: int


class BM25Index:
    """In-memory BM25 index with optional serialization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenize: TokenizeFn = default_tokenize):
        self.k1 = k1
        self.b = b
        self.tokenize = tokenize

        # Internal storage populated via build()
        self.postings: Dict[str, List[Posting]] = {}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self.idf: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def build(self, documents: Sequence[str]) -> None:
        """Build the index from an iterable of raw document strings."""
        postings: Dict[str, List[Posting]] = defaultdict(list)
        doc_len: List[int] = []
        doc_count = len(documents)
        term_doc_freq: Dict[str, int] = defaultdict(int)

        for doc_idx, text in enumerate(documents):
            tokens = self.tokenize(text)
            doc_length = len(tokens)
            doc_len.append(doc_length)
            if not tokens:
                continue
            freq = Counter(tokens)
            for term, term_freq in freq.items():
                postings[term].append(Posting(doc_idx, term_freq))
            for term in freq.keys():
                term_doc_freq[term] += 1

        avgdl = sum(doc_len) / doc_count if doc_count else 0.0

        idf: Dict[str, float] = {}
        for term, df in term_doc_freq.items():
            idf[term] = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))

        self.postings = dict(postings)
        self.doc_len = doc_len
        self.avgdl = avgdl
        self.doc_count = doc_count
        self.idf = idf

    # ------------------------------------------------------------------
    def score(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Return top_k document indices with BM25 scores for the query."""
        if not self.postings:
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term)
            if idf is None:
                continue
            for posting in postings:
                doc_idx = posting.doc_id
                freq = posting.term_freq
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += idf * freq * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialize the index to disk."""
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
            "idf": self.idf,
            # Convert postings to primitive data for pickle stability
            "postings": {term: [(p.doc_id, p.term_freq) for p in plist] for term, plist in self.postings.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path, tokenize: TokenizeFn = default_tokenize) -> "BM25Index":
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        index = cls(k1=payload["k1"], b=payload["b"], tokenize=tokenize)
        index.doc_len = payload["doc_len"]
        index.avgdl = payload["avgdl"]
        index.doc_count = payload["doc_count"]
        index.idf = payload["idf"]
        index.postings = {
            term: [Posting(doc_id=doc_id, term_freq=tf) for doc_id, tf in plist]
            for term, plist in payload["postings"].items()
        }
        return index


__all__ = ["BM25Index", "default_tokenize"]
