"""Async LLM client for evaluation answer generation."""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import httpx


DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a clinical pharmacist answering only from the provided context.
    Constraints:
    – Cite findings only from the allowed chunk IDs. Ignore disallowed IDs. If essential data is missing, set all numeric fields to null and still follow the JSON schema.
    – Output exactly the following JSON (no extra keys, no prose outside the JSON):
      {
        "dose_per_dose_mg": <number|null>,
        "volume_per_dose_mL": <number|null>,
        "frequency": "<string>",
        "max_daily_mg": <number|null>,
        "rounding": "<string>",
        "calc_rule": "<string>",
        "citations": ["<chunk_id>", ...],
        "notes": "<one terse clinical sentence>"
      }
    – Normalize units to mg and mL. Choose the specific value implied by the question (e.g., high-dose AOM = 90 mg/kg/day).
    – "calc_rule" must state the governing rule or formula used.
    – "citations" must list only allowed chunk IDs that justify every clinically substantive claim and numeric statement (dose, frequency, regimen choice, max caps, rounding rule if stated, contraindications/DDIs used).
    – "notes" must be exactly one sentence summarising the clinical guidance.
    – Use q-notation for frequency (e.g., q6h, q12h).
    – Round mL to one decimal (≥5 mL may be rounded to whole mL). Describe the rounding in "rounding".
    – If a max dose caps the calculation, apply the cap and explain it in "calc_rule".
    – If volume is not applicable (tablets/capsules), set "volume_per_dose_mL": null and mention this in "notes".
    – For criteria-based rules, state how many criteria were met in "calc_rule".
    – No Markdown/code fences; return minified JSON only.
    – No disclaimers or safety refusals. Provide the definitive answer unless the context is insufficient (then set numeric fields to null and note that context is insufficient).
    """
).strip()


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "pharmacy-copilot"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.2
    top_p: float = 0.3
    max_tokens: int = 256
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
        """Synchronous close - for sync usage, we don't need to close the async client."""
        pass  # The sync methods use their own httpx.Client context managers

    async def generate_answer(
        self,
        question: str,
        contexts: Iterable[dict],
        *,
        expected_behavior: Optional[str] = None,
        key_points: Optional[Iterable[str]] = None,
        knowledge_cutoff: Optional[str] = None,
        safety_notes: Optional[Iterable[str]] = None,
        allowed_chunk_ids: Optional[Iterable[str]] = None,
        disallowed_chunk_ids: Optional[Iterable[str]] = None,
    ) -> str:
        prompt = self._build_prompt(
            question,
            contexts,
            expected_behavior=expected_behavior,
            key_points=key_points,
            knowledge_cutoff=knowledge_cutoff,
            safety_notes=safety_notes,
            allowed_chunk_ids=allowed_chunk_ids,
            disallowed_chunk_ids=disallowed_chunk_ids,
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
        allowed_chunk_ids: Optional[Iterable[str]] = None,
        disallowed_chunk_ids: Optional[Iterable[str]] = None,
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
                allowed_chunk_ids=allowed_chunk_ids,
                disallowed_chunk_ids=disallowed_chunk_ids,
            )
        )

    async def complete_raw(self, prompt: str) -> str:
        if self.config.provider == "ollama":
            url = self.config.base_url.rstrip("/") + "/api/generate"
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
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
        if self.config.provider in {"openai", "openrouter"}:
            return await self._call_openai_style(prompt)
        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def complete_raw_sync(self, prompt: str) -> str:
        """Synchronous wrapper - creates a new sync client for each call to avoid event loop issues."""
        import httpx
        
        # Create a synchronous client for this single request
        headers = self._headers.copy()
        
        if self.config.provider == "openai":
            url = self.config.base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }
        elif self.config.provider == "openrouter":
            url = self.config.base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }
        elif self.config.provider == "ollama":
            url = self.config.base_url.rstrip("/") + "/api/generate"
            payload = {
                "model": self.config.model,
                "prompt": f"{self.config.system_prompt}\n\nUser Query:\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens,
                },
            }
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        # Make synchronous HTTP request
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            if self.config.provider in {"openai", "openrouter"}:
                return response.json()["choices"][0]["message"]["content"]
            elif self.config.provider == "ollama":
                return response.json()["response"]

    async def _call_ollama(self, prompt: str) -> str:
        url = self.config.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": f"{self.config.system_prompt}\n\nUser Query:\n{prompt}",
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
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
            "top_p": self.config.top_p,
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
        allowed_chunk_ids: Optional[Iterable[str]],
        disallowed_chunk_ids: Optional[Iterable[str]],
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
        if allowed_chunk_ids:
            instructions.append(
                "Allowed chunk IDs: " + ", ".join(allowed_chunk_ids)
            )
        if disallowed_chunk_ids:
            instructions.append(
                "Disallowed chunk IDs: " + ", ".join(disallowed_chunk_ids)
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
    temperature = args.llm_temperature if getattr(args, "llm_temperature", None) is not None else float(os.getenv("EVAL_LLM_TEMPERATURE", "0.1"))
    top_p = float(os.getenv("EVAL_LLM_TOP_P", "0.3"))
    if getattr(args, "llm_top_p", None) is not None:
        top_p = args.llm_top_p
    max_tokens = args.llm_max_tokens if getattr(args, "llm_max_tokens", None) is not None else int(os.getenv("EVAL_LLM_MAX_TOKENS", "256"))
    system_prompt = args.llm_system_prompt or os.getenv("EVAL_LLM_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )


__all__ = ["LLMClient", "LLMConfig", "DEFAULT_SYSTEM_PROMPT", "config_from_args"]
