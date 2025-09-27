#!/usr/bin/env python3
"""Section-aware ingestion utility for Lexicomp drug monographs.

Parses HTML monographs, extracts metadata + sections, and emits
JSON suitable for downstream retrieval experiments.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DIV_OPEN = "<div"
DIV_CLOSE = "</div>"
DRUG_SECTION_MARKER = "drugH1Div"


@dataclass
class Section:
    order: int
    section_id: str
    section_title: str
    section_code: Optional[str]
    class_attr: str
    raw_html: str
    text: str
    chunks: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class Monograph:
    doc_id: str
    path: Path
    drug_title: Optional[str]
    source_url: Optional[str]
    lexicomp_global_id: Optional[str]
    topic_version: Optional[str]
    copyright_notice: Optional[str]
    intro_html: Optional[str]
    intro_text: Optional[str]
    sections: List[Section]
    references: List[Dict[str, object]]


def _slice_balanced_block(block: str, start: int, tag: str) -> str:
    """Return substring spanning a <tag>...</tag> block with proper nesting."""
    open_token = f"<{tag}"
    close_token = f"</{tag}>"
    if not block.startswith(open_token, start):
        raise ValueError(f"Expected <{tag} at start index")
    depth = 0
    pos = start
    end = len(block)
    while pos < end:
        if block.startswith(open_token, pos):
            depth += 1
            pos = block.find('>', pos)
            if pos == -1:
                raise ValueError(f"Malformed <{tag}> start tag")
            pos += 1
            continue
        if block.startswith(close_token, pos):
            pos = block.find('>', pos)
            if pos == -1:
                raise ValueError(f"Malformed </{tag}> end tag")
            pos += 1
            depth -= 1
            if depth == 0:
                return block[start:pos]
            continue
        pos += 1
    raise ValueError(f"Unbalanced <{tag}> block")


def _html_to_text(html_fragment: str) -> str:
    """Convert a small HTML snippet to plain text, keeping bullets and spacing."""
    text = html_fragment
    # Normalize block-level tags to newlines to preserve structure.
    text = re.sub(r"</?(p|div|li|tr|h[0-9])[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Drop remaining tags.
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    # Collapse whitespace, but keep intentional newlines.
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


def _chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    """Split text into roughly max_chars-sized chunks at paragraph boundaries."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current and current_len + para_len + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _extract_meta(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.S)
    return unescape(match.group(1).strip()) if match else None


def parse_monograph(path: Path) -> Monograph:
    html = path.read_text(encoding="utf-8")
    doc_id = path.stem

    drug_title = _extract_meta(r"<div id=\"drugTitle\">(.*?)</div>", html)
    source_url = _extract_meta(r"<meta name=\"source\" content=\"(.*?)\"", html)
    lexicomp_global_id = _extract_meta(r"lco/action/api/find/globalid/(\d+)", html)
    topic_version = _extract_meta(r"<div id=\"topicVersionRevision\">(.*?)</div>", html)
    copyright_notice = _extract_meta(r"<div id=\"drugCopy\">(.*?)</div>", html)

    topic_text_start = html.find('<div id="topicText"')
    if topic_text_start == -1:
        raise ValueError(f"topicText not found in {path}")
    topic_text_block = _slice_balanced_block(html, topic_text_start, "div")
    inner_start = topic_text_block.find('>')
    topic_inner = topic_text_block[inner_start + 1 : -len(DIV_CLOSE)]

    sections: List[Section] = []
    intro_html = None
    intro_text = None

    # Grab intro text before first drugH1Div.
    first_section_idx = topic_inner.find(DRUG_SECTION_MARKER)
    if first_section_idx != -1:
        first_div_start = topic_inner.rfind('<div', 0, first_section_idx)
        intro_html = topic_inner[:first_div_start] if first_div_start != -1 else None
    else:
        intro_html = topic_inner
    if intro_html:
        intro_html = intro_html.strip()
    if intro_html:
        intro_text = _html_to_text(intro_html)
        if intro_text:
            intro_text = intro_text.strip()
        else:
            intro_text = None
    else:
        intro_html = None

    pos = 0
    order = 0
    seen_positions = set()
    while True:
        marker_idx = topic_inner.find(DRUG_SECTION_MARKER, pos)
        if marker_idx == -1:
            break
        div_start = topic_inner.rfind('<div', pos, marker_idx)
        if div_start == -1:
            break
        if div_start in seen_positions:
            pos = marker_idx + len(DRUG_SECTION_MARKER)
            continue
        seen_positions.add(div_start)
        block_html = _slice_balanced_block(topic_inner, div_start, "div")
        pos = div_start + len(block_html)

        start_tag_end = block_html.find('>')
        start_tag = block_html[: start_tag_end + 1]
        class_attr = _extract_meta(r'class="([^"]+)"', start_tag)
        section_id = _extract_meta(r'id="([^"]+)"', start_tag) or f"{doc_id}_section_{order}"
        section_title = _extract_meta(r'<span class="drugH1">(.*?)</span>', block_html) or ""
        section_title = section_title.strip()
        section_code = None
        if class_attr:
            parts = [p for p in class_attr.split() if p not in {"block", "list", "ex_sect_xr", "thclist", "drugH1Div", "drugBrandNames"}]
            section_code = parts[0] if parts else None
        inner_html = block_html[start_tag_end + 1 : -len(DIV_CLOSE)]
        section_text = _html_to_text(inner_html)
        chunk_texts = _chunk_text(section_text)
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append(
                {
                    "chunk_index": idx,
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "section_title": section_title,
                    "text": chunk_text,
                    "char_length": len(chunk_text),
                }
            )
        sections.append(
            Section(
                order=order,
                section_id=section_id,
                section_title=section_title,
                section_code=section_code,
                class_attr=class_attr or "",
                raw_html=block_html,
                text=section_text,
                chunks=chunks,
            )
        )
        order += 1

    references: List[Dict[str, object]] = []
    references_start = html.find('<div class="headingAnchor" id="references">')
    if references_start != -1:
        ref_block = html[references_start:]
        for match in re.finditer(r'<div class="reference">(.*?)</div>', ref_block, flags=re.S):
            ref_html = match.group(1)
            ref_text = _html_to_text(ref_html)
            pubmed_match = re.search(r'PubMed <a .*?>(\d+)</a>', ref_html)
            pubmed_id = pubmed_match.group(1) if pubmed_match else None
            references.append(
                {
                    "html": ref_html.strip(),
                    "text": ref_text,
                    "pubmed_id": pubmed_id,
                }
            )
    return Monograph(
        doc_id=doc_id,
        path=path,
        drug_title=drug_title,
        source_url=source_url,
        lexicomp_global_id=lexicomp_global_id,
        topic_version=topic_version,
        copyright_notice=copyright_notice,
        intro_html=intro_html,
        intro_text=intro_text,
        sections=sections,
        references=references,
    )


def serialize_monograph(mono: Monograph) -> Dict[str, object]:
    data = {
        "doc_id": mono.doc_id,
        "drug_title": mono.drug_title,
        "source_url": mono.source_url,
        "lexicomp_global_id": mono.lexicomp_global_id,
        "topic_version": mono.topic_version,
        "copyright_notice": mono.copyright_notice,
        "intro": {
            "html": mono.intro_html,
            "text": mono.intro_text,
        }
        if mono.intro_html or mono.intro_text
        else None,
        "sections": [
            {
                "order": section.order,
                "section_id": section.section_id,
                "section_title": section.section_title,
                "section_code": section.section_code,
                "class": section.class_attr,
                "text": section.text,
                "chunks": section.chunks,
            }
            for section in mono.sections
        ],
        "references": mono.references,
    }
    return data


def iter_monographs(input_dir: Path, limit: Optional[int] = None) -> Iterable[Monograph]:
    files = sorted(input_dir.glob('*.html'))
    for idx, path in enumerate(files):
        if limit is not None and idx >= limit:
            break
        yield parse_monograph(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Lexicomp HTML monographs")
    parser.add_argument("input_dir", type=Path, help="Directory containing monograph HTML files")
    parser.add_argument("output", type=Path, help="Path to output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()

    monographs = [serialize_monograph(mono) for mono in iter_monographs(args.input_dir, args.limit)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(monographs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
