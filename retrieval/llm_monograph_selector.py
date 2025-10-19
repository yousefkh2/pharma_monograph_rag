from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from retrieval.bm25_index import default_tokenize
from retrieval.search_utils import canonicalize_drug_title, load_metadata


@dataclass
class Monograph:
    doc_id: str
    drug_title: str
    tokens: frozenset[str]
    chunks: List[dict]

    @property
    def display_name(self) -> str:
        return self.drug_title or self.doc_id.replace("-", " ")


class MonographCatalog:
    """Collection of Lexicomp monographs with chunk access helpers."""

    def __init__(self, monographs: Sequence[Monograph]):
        self._monographs: List[Monograph] = list(monographs)
        self._by_doc_id: Dict[str, Monograph] = {mono.doc_id: mono for mono in monographs}

    @classmethod
    def from_metadata(cls, records: Sequence[dict]) -> "MonographCatalog":
        chunks_by_doc: Dict[str, List[dict]] = {}
        titles: Dict[str, str] = {}
        for record in records:
            doc_id = record.get("doc_id")
            if not doc_id:
                continue
            chunks_by_doc.setdefault(doc_id, []).append(record)
            if doc_id not in titles:
                titles[doc_id] = record.get("drug_title") or doc_id

        monographs: List[Monograph] = []
        for doc_id, chunks in chunks_by_doc.items():
            title = titles.get(doc_id, doc_id)
            canonical = canonicalize_drug_title(title)
            token_set = set(default_tokenize(canonical))
            if not token_set:
                token_set = set(default_tokenize(doc_id))
            monographs.append(
                Monograph(
                    doc_id=doc_id,
                    drug_title=title,
                    tokens=frozenset(token_set),
                    chunks=sorted(chunks, key=lambda item: item.get("chunk_id") or ""),
                )
            )

        monographs.sort(key=lambda item: item.display_name.lower())
        return cls(monographs)

    @classmethod
    def from_metadata_path(cls, metadata_path: str | Path) -> "MonographCatalog":
        records = load_metadata(Path(metadata_path))
        return cls.from_metadata(records)

    def __iter__(self) -> Iterable[Monograph]:
        return iter(self._monographs)

    def __len__(self) -> int:
        return len(self._monographs)

    def get(self, doc_id: str) -> Optional[Monograph]:
        return self._by_doc_id.get(doc_id)

    def build_catalog_block(
        self,
        candidates: Optional[Sequence[Monograph]] = None,
        *,
        snippet_map: Optional[Dict[str, str]] = None,
    ) -> str:
        items = list(candidates) if candidates is not None else self._monographs
        lines: List[str] = []
        for idx, mono in enumerate(items, start=1):
            snippet = ""
            if snippet_map:
                snippet = snippet_map.get(mono.doc_id, "") or ""
            snippet = snippet.strip().replace("\n", " ")
            if snippet and len(snippet) > 160:
                snippet = snippet[:160].rstrip() + "…"
            if snippet:
                lines.append(f"{idx}. {mono.display_name} (doc_id={mono.doc_id}) – {snippet}")
            else:
                lines.append(f"{idx}. {mono.display_name} (doc_id={mono.doc_id})")
        return "\n".join(lines)

    def score_overlap(self, tokens: Iterable[str]) -> Optional[Monograph]:
        needle = set(tokens)
        best: Optional[Monograph] = None
        best_overlap = 0
        for mono in self._monographs:
            overlap = len(needle & mono.tokens)
            if overlap > best_overlap:
                best = mono
                best_overlap = overlap
        return best if best_overlap else None

    def top_k_by_token_overlap(self, tokens: Iterable[str], k: int) -> List[Monograph]:
        needle = set(tokens)
        scored: List[tuple[int, Monograph]] = []
        for mono in self._monographs:
            overlap = len(needle & mono.tokens)
            if overlap:
                scored.append((overlap, mono))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1].display_name.lower()))
        return [mono for _, mono in scored[: max(1, k)]]

    def build_context(self, doc_id: str, include_text: bool = True) -> List[dict]:
        monograph = self.get(doc_id)
        if not monograph:
            return []
        context: List[dict] = []
        for rank, chunk in enumerate(monograph.chunks, start=1):
            text = chunk.get("text") or ""
            snippet = text.replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:240].rstrip() + "…"
            entry = {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": monograph.doc_id,
                "drug_title": monograph.drug_title,
                "section_title": chunk.get("section_title"),
                "section_code": chunk.get("section_code"),
                "snippet": snippet,
            }
            if include_text:
                entry["text"] = text
            context.append(entry)
        return context


@dataclass
class MonographSelection:
    doc_id: Optional[str]
    raw_response: str
    prompt: str
    reason: Optional[str] = None
    confidence: Optional[float] = None


DEFAULT_SELECTOR_PROMPT = (
    "You are an expert pharmacy monograph selector. Choose exactly one doc_id from the catalog that best answers the question. "
    "Rules: (1) Match the exact drug entity/formulation implied; prefer single-drug monographs unless the question clearly requests a combo. "
    "(2) Respect negations/exclusions (e.g., 'NOT amoxicillin-clavulanate' means never select that doc). "
    "(3) Align with clinical context: patient age group, renal/hepatic status, route, indication, dosing intent. "
    "(4) Normalize abbreviations and noisy spellings (AOM, NVAF, BBW, mg/kg/day, brand names). "
    "(5) If multiple docs are related, pick the most specific one that directly answers the question. "
    'Respond only with compact JSON: {{"doc_id":"<doc_id>","confidence":<0-1>,"reason":"<15-40 words>"}}. '
    "Do not include any additional text.\n\n"
    "Question:\n{question}\n\n"
    "Monograph catalog:\n{catalog}\n"
)


class LLMMonographSelector:
    """LLM-driven selector that picks a monograph by doc_id."""

    def __init__(
        self,
        catalog: MonographCatalog,
        llm_client,
        *,
        prompt_template: str = DEFAULT_SELECTOR_PROMPT,
        max_catalog_items: int = 30,
    ) -> None:
        self.catalog = catalog
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.max_catalog_items = max(1, max_catalog_items)

    async def select(
        self,
        question: str,
        *,
        candidates: Optional[Sequence[Monograph]] = None,
        snippet_map: Optional[Dict[str, str]] = None,
    ) -> MonographSelection:
        candidate_list = [mono for mono in candidates or [] if mono]
        if not candidate_list:
            tokens = default_tokenize(question)
            candidate_list = self.catalog.top_k_by_token_overlap(tokens, self.max_catalog_items)
        if not candidate_list:
            candidate_list = list(self.catalog)[: self.max_catalog_items]

        catalog_block = self.catalog.build_catalog_block(candidate_list, snippet_map=snippet_map)
        prompt = self.prompt_template.format(question=question.strip(), catalog=catalog_block)
        raw_response = await self.llm_client.complete_raw(prompt)
        doc_id, confidence, reason = self._parse_response(raw_response)
        if doc_id is None or not self.catalog.get(doc_id):
            fallback = self._fallback_selection(question, raw_response, candidate_list)
            if fallback:
                doc_id = fallback.doc_id
            else:
                doc_id = None
        return MonographSelection(
            doc_id=doc_id,
            raw_response=raw_response,
            prompt=prompt,
            reason=reason,
            confidence=confidence,
        )

    def _parse_response(self, response: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
        text = response.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            fragment = match.group(0)
            try:
                data = json.loads(fragment)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                doc_id = data.get("doc_id") or data.get("monograph_id")
                confidence = None
                raw_conf = data.get("confidence")
                if isinstance(raw_conf, (int, float)):
                    confidence = float(raw_conf)
                elif isinstance(raw_conf, str):
                    try:
                        confidence = float(raw_conf)
                    except ValueError:
                        confidence = None
                reason = data.get("reason") if isinstance(data.get("reason"), str) else None
                if doc_id:
                    doc_id = doc_id.strip()
                return doc_id, confidence, reason
        # Fallback: try to spot doc_id token explicitly mentioned
        for mono in self.catalog:
            if mono.doc_id in text:
                return mono.doc_id, None, None
        return None, None, None

    def _fallback_selection(
        self,
        question: str,
        response: str,
        candidates: Sequence[Monograph],
    ) -> Optional[Monograph]:
        tokens = set(default_tokenize(question)) | set(default_tokenize(response))
        pool = candidates if candidates else list(self.catalog)
        best: Optional[Monograph] = None
        best_overlap = 0
        for mono in pool:
            overlap = len(tokens & mono.tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = mono
        if best:
            return best
        return self.catalog.score_overlap(tokens)


__all__ = [
    "Monograph",
    "MonographCatalog",
    "MonographSelection",
    "LLMMonographSelector",
]
