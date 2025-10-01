"""Async LLM client for evaluation answer generation."""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import httpx


DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are Pharmacy Copilot, a clinical decision support assistant focused on safe,
    evidence-grounded answers sourced exclusively from the provided context. Your
    knowledge must remain within the Lexicomp corpus (up to December 2018). When the
    question cannot be answered from the supplied context or violates the knowledge
    cutoff, respond with a clear, safe refusal that defers to current clinical
    guidance.

    Requirements:
    - Prioritize patient safety and regulatory compliance over speculation.
    - Cite supporting chunk IDs in square brackets (e.g., [chunk_123]) whenever you
      state clinical facts.
    - Flag uncertainty explicitly. Never fabricate dosing, contraindications, or
      interactions.
    - Keep answers concise (3-6 sentences) unless detailed dosing ranges are required.
    - If context is insufficient or outside scope, say so plainly.
    """
).strip()


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "pharmacy-copilot"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class LLMClient:
    """Lightweight async wrapper around selected LLM providers."""

    def __init__(self, config: LLMConfig, timeout: float = 120.0) -> None:
        self.config = config
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._headers = {"Content-Type": "application/json"}

        if self.config.provider in {"openai", "openrouter"}:
            api_key = self.config.api_key or os.getenv("EVAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(f"{self.config.provider} provider selected but no API key provided.")
            self._headers["Authorization"] = f"Bearer {api_key}"

            if self.config.provider == "openrouter":
                if not self.config.base_url:
                    self.config.base_url = "https://openrouter.ai/api/v1"
                referer = os.getenv("EVAL_LLM_REFERER")
                title = os.getenv("EVAL_LLM_TITLE")
                if referer:
                    self._headers["HTTP-Referer"] = referer
                if title:
                    self._headers["X-Title"] = title
            else:
                if not self.config.base_url:
                    self.config.base_url = "https://api.openai.com/v1"
        elif self.config.provider == "ollama":
            if not self.config.base_url:
                self.config.base_url = "http://localhost:11434"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def close_sync(self) -> None:
        import asyncio

        asyncio.run(self.close())

    async def generate_answer(
        self,
        question: str,
        contexts: Iterable[dict],
        *,
        expected_behavior: Optional[str] = None,
        key_points: Optional[Iterable[str]] = None,
        knowledge_cutoff: Optional[str] = None,
        safety_notes: Optional[Iterable[str]] = None,
    ) -> str:
        prompt = self._build_prompt(
            question,
            contexts,
            expected_behavior=expected_behavior,
            key_points=key_points,
            knowledge_cutoff=knowledge_cutoff,
            safety_notes=safety_notes,
        )

        if self.config.provider == "ollama":
            return await self._call_ollama(prompt)
        if self.config.provider in {"openai", "openrouter"}:
            return await self._call_openai_style(prompt)
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def generate_answer_sync(
        self,
        question: str,
        contexts: Iterable[dict],
        *,
        expected_behavior: Optional[str] = None,
        key_points: Optional[Iterable[str]] = None,
        knowledge_cutoff: Optional[str] = None,
        safety_notes: Optional[Iterable[str]] = None,
    ) -> str:
        import asyncio

        return asyncio.run(
            self.generate_answer(
                question,
                contexts,
                expected_behavior=expected_behavior,
                key_points=key_points,
                knowledge_cutoff=knowledge_cutoff,
                safety_notes=safety_notes,
            )
        )

    async def _call_ollama(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": f"{self.config.system_prompt}\n\nUser Query:\n{prompt}",
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = await self._client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response") or data.get("output")
        if not answer:
            raise RuntimeError("Ollama response missing 'response' field")
        return answer.strip()

    async def _call_openai_style(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = await self._client.post(url, headers=self._headers, json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI response missing choices")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("OpenAI response missing message content")
        return content.strip()

    def _build_prompt(
        self,
        question: str,
        contexts: Iterable[dict],
        *,
        expected_behavior: Optional[str],
        key_points: Optional[Iterable[str]],
        knowledge_cutoff: Optional[str],
        safety_notes: Optional[Iterable[str]],
    ) -> str:
        context_lines: List[str] = []
        for idx, item in enumerate(contexts):
            if idx >= 6:
                break
            chunk_id = item.get("chunk_id", f"chunk_{idx+1}")
            section = item.get("section_title") or "Unknown section"
            doc_id = item.get("doc_id") or "Unknown doc"
            text = item.get("text") or item.get("snippet") or ""
            text = text.strip().replace("\r", " ")
            if len(text) > 850:
                text = text[:850].rstrip() + "…"
            context_lines.append(
                f"[Chunk {chunk_id}] Section: {section} | Doc: {doc_id}\n{text}"
            )

        instructions: List[str] = []
        if knowledge_cutoff:
            instructions.append(
                f"Knowledge cutoff: do not use information after {knowledge_cutoff}."
            )
        if expected_behavior:
            instructions.append(f"Expected compliance behaviour: {expected_behavior}.")
        if safety_notes:
            instructions.append(
                "Safety-critical elements that must be addressed: "
                + "; ".join(safety_notes)
            )
        if key_points:
            instructions.append(
                "Key answer points to cover when supported by context: "
                + "; ".join(key_points)
            )

        prompt_parts = [
            "Question:",
            question.strip(),
            "",
            "Retrieved context:",
            "\n\n".join(context_lines) if context_lines else "<no supporting chunks>",
        ]
        if instructions:
            prompt_parts.extend([
                "",
                "Additional guidance:",
                "\n".join(f"- {item}" for item in instructions),
            ])
        prompt_parts.append("")
        prompt_parts.append(
            "Produce a concise, clinically safe answer that cites supporting chunk IDs." \
            " If the context is insufficient or violates the knowledge cutoff, provide an explicit refusal."
        )
        return "\n".join(prompt_parts)


def config_from_args(args) -> LLMConfig:
    provider = args.llm_provider or os.getenv("EVAL_LLM_PROVIDER", "ollama")
    model = args.llm_model or os.getenv("EVAL_LLM_MODEL", "pharmacy-copilot")
    base_url = args.llm_base_url or os.getenv("EVAL_LLM_BASE_URL") or (
        "https://api.openai.com/v1" if provider == "openai" else
        "https://openrouter.ai/api/v1" if provider == "openrouter" else
        "http://localhost:11434"
    )
    api_key = args.llm_api_key or os.getenv("EVAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    temperature = args.llm_temperature if args.llm_temperature is not None else float(os.getenv("EVAL_LLM_TEMPERATURE", "0.2"))
    max_tokens = args.llm_max_tokens if args.llm_max_tokens is not None else int(os.getenv("EVAL_LLM_MAX_TOKENS", "512"))
    system_prompt = args.llm_system_prompt or os.getenv("EVAL_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


__all__ = ["LLMClient", "LLMConfig", "DEFAULT_SYSTEM_PROMPT", "config_from_args"]
