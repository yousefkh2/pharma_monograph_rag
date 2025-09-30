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
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

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
    """In-memory BM25 index with optional serialization and fuzzy matching."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenize: TokenizeFn = default_tokenize,
        enable_fuzzy: bool = True,
        similarity_threshold: float = 0.8,
        max_fuzzy_expansions: int = 3,
    ):
        self.k1 = k1
        self.b = b
        self.tokenize = tokenize
        self.enable_fuzzy = enable_fuzzy and RAPIDFUZZ_AVAILABLE
        self.similarity_threshold = similarity_threshold
        self.max_fuzzy_expansions = max_fuzzy_expansions

        # Internal storage populated via build()
        self.postings: Dict[str, List[Posting]] = {}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self.idf: Dict[str, float] = {}
        self.vocabulary: Set[str] = set()  # For fuzzy matching

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
        self.vocabulary = set(self.postings.keys())

    # ------------------------------------------------------------------
    def _get_fuzzy_expansions(self, term: str) -> List[Tuple[str, float]]:
        """Find similar terms in the vocabulary with their similarity scores.
        
        Returns a list of (term, similarity_weight) tuples where similarity_weight
        is between 0.0 and 1.0 based on the fuzzy match score.
        """
        if not self.enable_fuzzy or not RAPIDFUZZ_AVAILABLE or not self.vocabulary:
            return []
            
        # If exact match exists, no need for fuzzy matching
        if term in self.vocabulary:
            return []
            
        # Find similar terms using rapidfuzz
        matches = process.extract(
            term,
            self.vocabulary,
            scorer=fuzz.WRatio,  # Weighted ratio for better performance on different string lengths
            limit=self.max_fuzzy_expansions,
        )
        
        # Filter by similarity threshold and convert scores to weights
        expansions = []
        for match_result in matches:
            match_term, score = match_result[0], match_result[1]  # Handle tuple unpacking correctly
            # Filter out very short matches that might be spurious and check similarity threshold
            if (score >= (self.similarity_threshold * 100) and 
                len(match_term) >= 3 and  # Avoid very short spurious matches
                len(match_term) >= len(term) * 0.5):  # Avoid matches that are too short relative to query
                # Convert score to a weight between 0.0 and 1.0
                # Use a softer scaling to avoid over-weighting fuzzy matches
                similarity_weight = (score / 100.0) * 0.7  # Max weight is 0.7 for fuzzy matches
                expansions.append((match_term, similarity_weight))
        
        return expansions

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
            # Try exact match first
            postings = self.postings.get(term)
            idf = self.idf.get(term)
            
            if postings and idf is not None:
                # Process exact match with full weight
                for posting in postings:
                    doc_idx = posting.doc_id
                    freq = posting.term_freq
                    denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                    scores[doc_idx] += idf * freq * (self.k1 + 1) / denom
            else:
                # No exact match found, try fuzzy expansions
                fuzzy_expansions = self._get_fuzzy_expansions(term)
                for fuzzy_term, similarity_weight in fuzzy_expansions:
                    fuzzy_postings = self.postings.get(fuzzy_term)
                    fuzzy_idf = self.idf.get(fuzzy_term)
                    if fuzzy_postings and fuzzy_idf is not None:
                        for posting in fuzzy_postings:
                            doc_idx = posting.doc_id
                            freq = posting.term_freq
                            denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                            # Weight the contribution by similarity
                            fuzzy_score = fuzzy_idf * freq * (self.k1 + 1) / denom * similarity_weight
                            scores[doc_idx] += fuzzy_score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialize the index to disk."""
        payload = {
            "k1": self.k1,
            "b": self.b,
            "enable_fuzzy": self.enable_fuzzy,
            "similarity_threshold": self.similarity_threshold,
            "max_fuzzy_expansions": self.max_fuzzy_expansions,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "doc_count": self.doc_count,
            "idf": self.idf,
            "vocabulary": list(self.vocabulary),  # Convert set to list for serialization
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
            
        # Handle backward compatibility - older indices may not have fuzzy parameters
        index = cls(
            k1=payload["k1"],
            b=payload["b"],
            tokenize=tokenize,
            enable_fuzzy=payload.get("enable_fuzzy", True),
            similarity_threshold=payload.get("similarity_threshold", 0.8),
            max_fuzzy_expansions=payload.get("max_fuzzy_expansions", 3),
        )
        
        index.doc_len = payload["doc_len"]
        index.avgdl = payload["avgdl"]
        index.doc_count = payload["doc_count"]
        index.idf = payload["idf"]
        index.vocabulary = set(payload.get("vocabulary", payload["postings"].keys()))  # Backward compatibility
        index.postings = {
            term: [Posting(doc_id=doc_id, term_freq=tf) for doc_id, tf in plist]
            for term, plist in payload["postings"].items()
        }
        return index


__all__ = ["BM25Index", "default_tokenize"]
