from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal, Protocol

import anthropic
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .prompts import INGEST_SYSTEM_PROMPT

DEFAULT_INGEST_MODEL = "claude-opus-4-7"
MAX_OUTPUT_TOKENS = 4096
MESSAGE_TIMEOUT_S = 240

TEXT_RE = re.compile(r"\s+")


class SessionLike(Protocol):
    id: str
    timestamp: str
    messages: list[dict[str, Any]]


class EntityCandidate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: str = Field(default="unknown", min_length=1, max_length=80)

    @model_validator(mode="before")
    @classmethod
    def _coerce_entity(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value, "type": "unknown"}
        return value

    @field_validator("name", "type", mode="before")
    @classmethod
    def _strip_text(cls, value: Any, info: ValidationInfo) -> str:
        if value is None:
            return "unknown" if info.field_name == "type" else ""
        return TEXT_RE.sub(" ", str(value)).strip()


class FactCandidate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    polarity: Literal["+", "-"] = "+"
    entities: list[EntityCandidate] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("text", mode="before")
    @classmethod
    def _strip_fact_text(cls, value: Any) -> str:
        return TEXT_RE.sub(" ", str(value)).strip()

    @model_validator(mode="after")
    def _dedupe_entities(self) -> "FactCandidate":
        seen: set[str] = set()
        entities: list[EntityCandidate] = []
        for entity in self.entities:
            key = entity.name.casefold()
            if key and key not in seen:
                seen.add(key)
                entities.append(entity)
        self.entities = entities
        return self


class ExtractionResult(BaseModel):
    facts: list[FactCandidate] = Field(default_factory=list)


async def extract_facts(
    client: anthropic.AsyncAnthropic,
    session: SessionLike,
    *,
    model: str = DEFAULT_INGEST_MODEL,
) -> list[FactCandidate]:
    payload = {
        "session_id": session.id,
        "timestamp": session.timestamp,
        "turns": session.messages,
    }
    user_text = "EXTRACT MEMORY FACTS:\n" + json.dumps(payload, ensure_ascii=False)

    async with asyncio.timeout(MESSAGE_TIMEOUT_S):
        response = await client.messages.parse(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            output_config={"effort": "low"},
            output_format=ExtractionResult,
            system=[
                {
                    "type": "text",
                    "text": INGEST_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": user_text}]}
            ],
        )

    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"extraction hit max_tokens for session {session.id}")
    if response.stop_reason == "refusal":
        raise RuntimeError(f"extraction refused session {session.id}")

    parsed = _parsed_output(response)
    return _dedupe_facts(parsed.facts)


def _parsed_output(response: Any) -> ExtractionResult:
    for block in response.content:
        if getattr(block, "type", None) != "text":
            continue
        parsed = getattr(block, "parsed_output", None)
        if isinstance(parsed, ExtractionResult):
            return parsed
        if parsed is not None:
            return ExtractionResult.model_validate(parsed)
    raise ValueError("Anthropic structured extraction returned no parsed output")


def _dedupe_facts(facts: list[FactCandidate]) -> list[FactCandidate]:
    seen: set[str] = set()
    deduped: list[FactCandidate] = []
    for fact in facts:
        key = TEXT_RE.sub(" ", fact.text.casefold()).strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(fact)
    return deduped
