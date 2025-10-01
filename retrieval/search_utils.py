"""Shared search utilities and heuristics for Lexicomp BM25 retrieval."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from retrieval.bm25_index import BM25Index, default_tokenize

STOPWORDS: Set[str] = {
    "brand",
    "names",
    "us",
    "canada",
    "international",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "solution",
    "solutions",
    "oral",
    "injection",
    "injections",
    "cream",
    "spray",
    "patch",
    "patches",
    "gel",
    "ointment",
    "powder",
    "kit",
    "kits",
    "suspension",
    "dose",
    "doses",
    "dosing",
    "strength",
    "strengths",
    "mg",
    "mcg",
    "g",
    "ml",
    "units",
    "unit",
    "solution",
}


INTENT_SECTION_MAP = {
    "dose": ["dosing", "dosage", "dosage and administration", "administering"],
    "adjust": [
        "dosing",
        "dosage",
        "renal impairment",
        "renal dosing",
        "hepatic impairment",
        "hepatic dosing",
    ],
    "interactions": ["drug interactions", "interactions"],
    "contraindications": ["contraindications", "warnings", "warnings and precautions"],
    "pregnancy": ["pregnancy", "pregnancy considerations"],
    "lactation": ["lactation", "breastfeeding"],
}


ROUTE_KEYWORDS = {
    "oral",
    "tablet",
    "capsule",
    "iv",
    "intravenous",
    "im",
    "intramuscular",
    "subcutaneous",
    "sc",
    "topical",
    "ophthalmic",
    "otic",
    "inhalation",
    "nasal",
    "transdermal",
    "sublingual",
}


FORMULATION_KEYWORDS = {
    "solution",
    "suspension",
    "extended-release",
    "immediate-release",
    "er",
    "cr",
    "sr",
    "chewable",
    "elixir",
    "ointment",
    "cream",
    "gel",
}


SALT_KEYWORDS = {
    "acetate",
    "chloride",
    "hydrochloride",
    "phosphate",
    "succinate",
    "tartrate",
    "sodium",
    "potassium",
}


@dataclass
class DocumentInfo:
    chunk_indices: List[int]
    drug_title: Optional[str]
    canonical: str
    canonical_tokens: Set[str]
    synonyms: Set[str]


@dataclass
class DebugEntry:
    chunk_id: str
    doc_id: str
    section_title: Optional[str]
    base_score: float
    final_score: float
    adjustment: float
    reasons: List[str]


def canonicalize_drug_title(title: Optional[str]) -> str:
    if not title:
        return ""
    base = title.split(":")[0]
    base = base.lower()
    base = re.sub(r"[^a-z0-9\s-]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base


def load_metadata(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_doc_index(metadata: Sequence[dict]) -> Dict[str, DocumentInfo]:
    doc_index: Dict[str, DocumentInfo] = {}
    for idx, record in enumerate(metadata):
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        entry = doc_index.get(doc_id)
        if not entry:
            canonical = canonicalize_drug_title(record.get("drug_title"))
            entry = DocumentInfo(
                chunk_indices=[],
                drug_title=record.get("drug_title"),
                canonical=canonical,
                canonical_tokens=set(default_tokenize(canonical)),
                synonyms=set(),
            )
            doc_index[doc_id] = entry
        entry.chunk_indices.append(idx)
        section_title = (record.get("section_title") or "").lower()
        if "brand names" in section_title:
            tokens = default_tokenize(record.get("text", ""))
            for token in tokens:
                if token in STOPWORDS or len(token) < 3:
                    continue
                entry.synonyms.add(token)
        # Always include doc_id tokens as synonyms for matching (eg, hyphenated names)
        for token in doc_id.split('-'):
            clean = re.sub(r"[^a-z0-9]", "", token.lower())
            if clean:
                entry.synonyms.add(clean)
    return doc_index


def detect_target_doc_ids(query: str, doc_index: Dict[str, DocumentInfo]) -> Set[str]:
    query_lower = query.lower()
    query_tokens = set(default_tokenize(query))
    
    # Intent keywords that don't count as drug names
    intent_keywords = {
        "interactions", "interact", "interaction", "interacts",
        "dose", "dosing", "dosage", "doses",
        "contraindications", "contraindication", "avoid",
        "pregnancy", "pregnant", "lactation", "breastfeeding",
        "side", "effects", "adverse", "reactions",
        "warnings", "precautions", "cautions"
    }
    
    # Filter out intent keywords to get actual drug tokens
    drug_tokens = {token for token in query_tokens if token not in intent_keywords and token not in STOPWORDS}
    single_drug_query = len(drug_tokens) == 1
    matches: Set[str] = set()
    for doc_id, info in doc_index.items():
        if info.canonical and info.canonical in query_lower:
            matches.add(doc_id)
            continue
        if info.canonical_tokens and info.canonical_tokens.issubset(query_tokens):
            matches.add(doc_id)
            continue
        if info.synonyms and query_tokens.intersection(info.synonyms):
            # For single-drug queries, avoid picking combination products that contain " and " or "/" in their title
            title = info.drug_title.lower() if info.drug_title else ""
            if single_drug_query and (" and " in title or "/" in title or "," in title):
                continue
            matches.add(doc_id)
            continue
    return matches


def infer_intents(query_tokens: Sequence[str], query_text: str) -> Dict[str, bool]:
    stems = {token for token in query_tokens}
    query_lower = query_text.lower()
    dose_terms = {token for token in stems if token.startswith("dose") or token.startswith("dosing")}
    adjust_terms = {token for token in stems if token.startswith("adjust") or token.startswith("titr")}
    inr = "inr" in stems
    interaction_terms = {token for token in stems if token.startswith("interact")}
    contraindication_terms = {token for token in stems if token.startswith("contraind") or token == "avoid"}
    pregnancy = "pregnancy" in query_lower or "pregnant" in stems
    lactation = "lactation" in query_lower or "breastfeeding" in query_lower or "breast" in stems
    guideline = "guideline" in query_lower or "guidelines" in query_lower or "recommendation" in query_lower

    routes = {keyword for keyword in ROUTE_KEYWORDS if keyword in query_lower}
    formulations = {keyword for keyword in FORMULATION_KEYWORDS if keyword in query_lower}

    return {
        "dose": bool(dose_terms),
        "adjust": bool(adjust_terms or "adjustment" in query_lower),
        "inr": inr,
        "dose_adjust_phrase": "dose adjustment" in query_lower,
        "interactions": bool(interaction_terms),
        "contraindications": bool(contraindication_terms),
        "pregnancy": pregnancy,
        "lactation": lactation,
        "guideline": guideline,
        "routes": routes,
        "formulations": formulations,
    }


def _resolve_intended_sections(intents: Dict[str, bool]) -> List[str]:
    intended: List[str] = []
    for intent_key, sections in INTENT_SECTION_MAP.items():
        if intents.get(intent_key):
            intended.extend(sections)
    return intended


def _apply_rerank(
    results: Sequence[Tuple[int, float]],
    metadata: Sequence[dict],
    doc_index: Dict[str, DocumentInfo],
    query: str,
    target_doc_ids: Set[str],
    intents: Dict[str, bool],
    top_k: int,
    debug: bool,
) -> Tuple[List[Tuple[int, float]], List[DebugEntry]]:
    # Map detected intents (dosing, interactions, pregnancy, etc.) to section hints and
    # metadata keywords so we can tilt the hybrid score toward clinically appropriate chunks.
    intended_sections = _resolve_intended_sections(intents)
    routes = intents.get("routes") or set()
    formulations = intents.get("formulations") or set()
    query_token_set = set(default_tokenize(query))
    
    # Intent keywords that don't count as drug names
    intent_keywords = {
        "interactions", "interact", "interaction", "interacts",
        "dose", "dosing", "dosage", "doses",
        "contraindications", "contraindication", "avoid",
        "pregnancy", "pregnant", "lactation", "breastfeeding",
        "side", "effects", "adverse", "reactions",
        "warnings", "precautions", "cautions"
    }
    
    # Filter out intent keywords to get actual drug tokens
    drug_tokens = {token for token in query_token_set if token not in intent_keywords and token not in STOPWORDS}
    single_drug_query = len(drug_tokens) == 1
    reranked: List[Tuple[int, float, float, List[str]]] = []
    for doc_idx, base_score in results:
        record = metadata[doc_idx]
        doc_id = record.get("doc_id")
        section_title = record.get("section_title") or ""
        text = record.get("text", "")
        text_lower = text.lower()
        section_lower = section_title.lower()

        adjustment = 0.0
        reasons: List[str] = []

        if target_doc_ids:
            if doc_id in target_doc_ids:
                adjustment += 6.0
                reasons.append("target-drug")
            else:
                adjustment -= 5.0
                reasons.append("off-drug")

        if single_drug_query:
            drug_title = (record.get("drug_title") or "").lower()
            if any(delim in drug_title for delim in {" and ", "/", ","}):
                adjustment -= 8.0  # Increased penalty for combination drugs in single-drug queries
                reasons.append("combo-penalty")
            elif query.lower().strip() in drug_title:
                # Boost exact drug name matches for single-drug queries
                adjustment += 4.0
                reasons.append("exact-drug-match")
        elif len(drug_tokens) >= 2:
            drug_title = (record.get("drug_title") or "").lower()
            missing = [token for token in drug_tokens if token not in drug_title]
            if missing:
                adjustment -= 4.0
                reasons.append("missing-combo-drug")

        if intended_sections:
            for keyword in intended_sections:
                if keyword and keyword in section_lower:
                    adjustment += 3.0
                    reasons.append(f"section-match:{keyword}")
                    break

        if routes:
            for route in routes:
                if route in section_lower or route in text_lower:
                    adjustment += 2.0
                    reasons.append(f"route-match:{route}")
                    break

        if formulations:
            for formulation in formulations:
                if formulation in section_lower or formulation in text_lower:
                    adjustment += 2.0
                    reasons.append(f"formulation-match:{formulation}")
                    break

        salt_hits = SALT_KEYWORDS.intersection(query_token_set)
        if salt_hits:
            drug_title = (record.get("drug_title") or "").lower()
            if any(salt in drug_title for salt in salt_hits):
                adjustment += 2.0
                reasons.append("salt-match")

        if intents.get("dose") or intents.get("adjust"):
            if section_lower.startswith("dosing"):
                adjustment += 5.0
                reasons.append("section-dosing")
                if section_lower.startswith("dosing: adult"):
                    adjustment += 1.5
                    reasons.append("dosing-adult-priority")
                elif "pediatric" in section_lower:
                    adjustment -= 1.0
                    reasons.append("dosing-pediatric-penalty")
                elif "geriatric" in section_lower:
                    adjustment -= 0.5
                    reasons.append("dosing-geriatric-penalty")
            elif "interaction" in section_lower:
                penalty = 3.0 if intents.get("adjust") else 2.0
                adjustment -= penalty
                reasons.append("section-interactions")
            elif "brand names" in section_lower:
                adjustment -= 2.5
                reasons.append("section-brand-names")
            elif "warning" in section_lower or "precaution" in section_lower:
                adjustment -= 2.0
                reasons.append("section-warnings")
            elif "pregnancy" in section_lower or "breast" in section_lower:
                adjustment -= 2.0
                reasons.append("section-pregnancy")
            else:
                adjustment -= 4.0
                reasons.append("non-dosing-section")
            if "not available in the us" in text_lower or "not available in us" in text_lower:
                adjustment -= 3.0
                reasons.append("availability-penalty")
            if "note:" in text_lower and "general" in text_lower:
                adjustment -= 3.0
                reasons.append("general-note-penalty")
        if intents.get("inr") and "inr" in text_lower:
            adjustment += 1.0
            reasons.append("inr-match")
        if intents.get("adjust") and ("adjust" in text_lower or "titr" in text_lower):
            adjustment += 0.8
            reasons.append("adjust-mention")
        if intents.get("dose") and "dose" in text_lower:
            adjustment += 0.5
            reasons.append("dose-mention")
        if intents.get("dose_adjust_phrase") and "dose adjustment" in text_lower:
            adjustment += 1.5
            reasons.append("dose-adjustment-phrase")
        if intents.get("interactions") and "interaction" in section_lower:
            adjustment += 2.5
            reasons.append("interaction-section")
        if intents.get("contraindications") and "contraindication" in section_lower:
            adjustment += 2.5
            reasons.append("contraindication-section")
        if intents.get("pregnancy") and "pregnancy" in section_lower:
            adjustment += 2.0
            reasons.append("pregnancy-section")
        if intents.get("lactation") and ("lactation" in section_lower or "breast" in section_lower):
            adjustment += 2.0
            reasons.append("lactation-section")
        if intents.get("guideline"):
            years = {int(year) for year in re.findall(r"(?:19|20)\d{2}", text_lower)}
            outdated = any(year < 2019 for year in years)
            if outdated:
                adjustment -= 3.0
                reasons.append("outdated-guideline")

        final_score = base_score + adjustment
        reranked.append((doc_idx, final_score, base_score, reasons))

    reranked.sort(key=lambda item: item[1], reverse=True)

    if target_doc_ids:
        target_items = [item for item in reranked if metadata[item[0]].get("doc_id") in target_doc_ids]
        if target_items:
            non_target_items = [item for item in reranked if metadata[item[0]].get("doc_id") not in target_doc_ids]
            reranked = target_items + non_target_items

    top_results = reranked[:top_k]

    debug_entries: List[DebugEntry] = []
    if debug:
        for doc_idx, final_score, base_score, reasons in top_results:
            record = metadata[doc_idx]
            debug_entries.append(
                DebugEntry(
                    chunk_id=record.get("chunk_id", ""),
                    doc_id=record.get("doc_id", ""),
                    section_title=record.get("section_title"),
                    base_score=base_score,
                    final_score=final_score,
                    adjustment=final_score - base_score,
                    reasons=reasons,
                )
            )

    return [(doc_idx, final_score) for doc_idx, final_score, _, _ in top_results], debug_entries


class BM25SearchEngine:
    """Wrapper around the BM25 index with domain-specific reranking."""

    def __init__(self, index: BM25Index, metadata: List[dict]):
        self.index = index
        self.metadata = metadata
        self.doc_index = build_doc_index(metadata)

    @classmethod
    def from_files(cls, index_path: Path, metadata_path: Path) -> "BM25SearchEngine":
        index = BM25Index.load(index_path, tokenize=default_tokenize)
        metadata = load_metadata(metadata_path)
        return cls(index, metadata)

    def _resolve_target_docs(self, query: str, must_drug: Optional[str]) -> Set[str]:
        if must_drug:
            needle = must_drug.lower()
            matches = {
                doc_id
                for doc_id, info in self.doc_index.items()
                if needle in doc_id.lower() or needle in (info.canonical or "")
            }
            return matches
        return detect_target_doc_ids(query, self.doc_index)

    def _make_snippet(self, text: str, limit: int = 240) -> str:
        snippet = text.replace("\n", " ")
        if len(snippet) > limit:
            snippet = snippet[:limit].rstrip() + "…"
        return snippet

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 200,
        must_drug: Optional[str] = None,
        debug: bool = False,
        include_text: bool = True,
    ) -> Tuple[List[dict], List[DebugEntry]]:
        initial_results = self.index.score(query, top_k=retrieve_k)
        if not initial_results:
            return [], []

        query_tokens = default_tokenize(query)
        intents = infer_intents(query_tokens, query)
        target_doc_ids = self._resolve_target_docs(query, must_drug)

        reranked, debug_entries = _apply_rerank(
            initial_results,
            self.metadata,
            self.doc_index,
            query,
            target_doc_ids,
            intents,
            top_k,
            debug,
        )

        hits: List[dict] = []
        for rank, (doc_idx, score) in enumerate(reranked, start=1):
            record = self.metadata[doc_idx]
            text = record.get("text", "")
            hits.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": record.get("chunk_id"),
                    "doc_id": record.get("doc_id"),
                    "drug_title": record.get("drug_title"),
                    "section_title": record.get("section_title"),
                    "section_code": record.get("section_code"),
                    "snippet": self._make_snippet(text),
                    "text": text if include_text else None,
                }
            )
        return hits, debug_entries


__all__ = [
    "BM25SearchEngine",
    "DebugEntry",
    "build_doc_index",
    "canonicalize_drug_title",
    "detect_target_doc_ids",
    "infer_intents",
    "load_metadata",
]
