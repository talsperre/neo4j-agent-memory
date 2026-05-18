from __future__ import annotations

import asyncio
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from openai import AsyncOpenAI

from .extract import EntityCandidate, FactCandidate

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
EMBED_BATCH_SIZE = 100
EMBED_TIMEOUT_S = 60
VECTOR_SUPERSESSION_THRESHOLD = 0.85

TOKEN_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
GENERIC_ENTITY_NAMES = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "user",
    "the user",
    "assistant",
    "chatgpt",
    "ai",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "user",
    "users",
    "was",
    "were",
    "with",
}


class SessionLike(Protocol):
    id: str
    timestamp: str


@dataclass(frozen=True)
class WriteStats:
    session_id: str
    extracted: int
    added: int
    superseded: int
    noop: int


@dataclass
class PreparedFact:
    id: str
    text: str
    polarity: str
    confidence: float
    entities: list[EntityCandidate]
    embedding: list[float]
    valid_from: datetime
    valid_to: datetime | None = None
    supersedes_id: str | None = None
    superseded_by_id: str | None = None


@dataclass
class ExistingFact:
    id: str
    text: str
    polarity: str
    entities: list[EntityCandidate]
    embedding: list[float] | None
    valid_from: datetime | None
    valid_to: datetime | None


@dataclass(frozen=True)
class Match:
    old: ExistingFact
    similarity: float
    entity_overlap: int
    token_overlap: float

    @property
    def score(self) -> tuple[float, float, int, float]:
        return (
            1.0 if self.similarity > VECTOR_SUPERSESSION_THRESHOLD else 0.0,
            self.similarity,
            self.entity_overlap,
            self.token_overlap,
        )


async def write_session_facts(
    driver: Any,
    openai_client: AsyncOpenAI,
    *,
    user_id: str,
    session: SessionLike,
    facts: list[FactCandidate],
) -> WriteStats:
    timestamp = _parse_timestamp(session.timestamp)
    embeddings = await _embed_facts(openai_client, facts)
    prepared = [
        PreparedFact(
            id=str(uuid.uuid4()),
            text=fact.text,
            polarity=fact.polarity,
            confidence=fact.confidence,
            entities=fact.entities,
            embedding=embedding,
            valid_from=timestamp,
        )
        for fact, embedding in zip(facts, embeddings, strict=True)
    ]

    async with driver.session() as neo4j_session:
        tx = await neo4j_session.begin_transaction()
        try:
            stats = await _reconcile_and_write(
                tx,
                user_id,
                session.id,
                timestamp,
                prepared,
                len(facts),
            )
        except Exception:
            await tx.rollback()
            raise
        await tx.commit()
        return stats


async def _embed_facts(
    openai_client: AsyncOpenAI, facts: list[FactCandidate]
) -> list[list[float]]:
    if not facts:
        return []

    embeddings: list[list[float]] = []
    for start in range(0, len(facts), EMBED_BATCH_SIZE):
        batch = facts[start : start + EMBED_BATCH_SIZE]
        async with asyncio.timeout(EMBED_TIMEOUT_S):
            response = await openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSIONS,
                input=[fact.text for fact in batch],
            )
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(batch):
            raise RuntimeError(
                f"embedding count mismatch: expected {len(batch)}, got {len(data)}"
            )
        embeddings.extend(_validate_embedding(item.embedding) for item in data)
    return embeddings


async def _reconcile_and_write(
    tx: Any,
    user_id: str,
    session_id: str,
    timestamp: datetime,
    facts: list[PreparedFact],
    extracted_count: int,
) -> WriteStats:
    await _merge_session(tx, user_id, session_id, timestamp)
    if not facts:
        return WriteStats(
            session_id=session_id,
            extracted=extracted_count,
            added=0,
            superseded=0,
            noop=0,
        )

    existing = await _fetch_existing_facts(tx, user_id)
    adds, supersedes, noop_count = _classify_facts(facts, existing, timestamp)
    all_new = adds + supersedes

    if all_new:
        await _create_facts(tx, user_id, session_id, timestamp, all_new)
    if supersedes:
        await _write_supersedes(tx, user_id, timestamp, supersedes)

    future_links = [fact for fact in adds if fact.superseded_by_id is not None]
    if future_links:
        await _write_future_supersedes(tx, user_id, future_links)

    return WriteStats(
        session_id=session_id,
        extracted=extracted_count,
        added=len(adds),
        superseded=len(supersedes),
        noop=noop_count,
    )


async def _merge_session(
    tx: Any, user_id: str, session_id: str, timestamp: datetime
) -> None:
    result = await tx.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (s:Session {user_id: $user_id, id: $session_id})
          ON CREATE SET s.timestamp = datetime($timestamp)
          ON MATCH SET s.timestamp = coalesce(s.timestamp, datetime($timestamp))
        MERGE (u)-[:HAS_SESSION]->(s)
        """,
        user_id=user_id,
        session_id=session_id,
        timestamp=_iso_z(timestamp),
    )
    await result.consume()


async def _fetch_existing_facts(tx: Any, user_id: str) -> list[ExistingFact]:
    result = await tx.run(
        """
        MATCH (:User {id: $user_id})-[:HAS_SESSION]->(s:Session {user_id: $user_id})-[:CONTAINS_FACT]->(f:Fact)
        WHERE f.user_id = $user_id
        OPTIONAL MATCH (f)-[:MENTIONS]->(e:Entity {user_id: $user_id})
        RETURN f.id AS id,
               f.text AS text,
               f.polarity AS polarity,
               f.embedding AS embedding,
               f.valid_from AS valid_from,
               f.valid_to AS valid_to,
               collect(CASE WHEN e.name IS NULL THEN NULL ELSE {name: e.name, type: e.type} END) AS entities
        """,
        user_id=user_id,
    )
    existing: list[ExistingFact] = []
    async for record in result:
        entities = [
            EntityCandidate.model_validate(entity)
            for entity in record["entities"]
            if entity is not None and entity.get("name")
        ]
        embedding = _optional_embedding(record["embedding"])
        existing.append(
            ExistingFact(
                id=record["id"],
                text=record["text"] or "",
                polarity=record["polarity"] or "+",
                entities=entities,
                embedding=embedding,
                valid_from=_coerce_datetime(record["valid_from"]),
                valid_to=_coerce_datetime(record["valid_to"]),
            )
        )
    return existing


def _classify_facts(
    facts: list[PreparedFact],
    existing: list[ExistingFact],
    timestamp: datetime,
) -> tuple[list[PreparedFact], list[PreparedFact], int]:
    adds: list[PreparedFact] = []
    supersedes: list[PreparedFact] = []
    noop_count = 0

    for fact in facts:
        matches = sorted(
            (_match for old in existing if (_match := _match_fact(fact, old))),
            key=lambda match: match.score,
            reverse=True,
        )
        active_matches = [
            match for match in matches if _is_active_at(match.old, timestamp)
        ]

        noop = next(
            (match for match in active_matches if _is_noop(fact, match)),
            None,
        )
        if noop is not None:
            noop_count += 1
            continue

        if active_matches:
            chosen = active_matches[0]
            fact.supersedes_id = chosen.old.id
            fact.valid_to = chosen.old.valid_to
            chosen.old.valid_to = timestamp
            supersedes.append(fact)
            existing.append(_as_existing(fact))
            continue

        future_match = next(
            (
                match
                for match in matches
                if match.old.valid_from is not None and match.old.valid_from > timestamp
            ),
            None,
        )
        if future_match is not None:
            fact.valid_to = future_match.old.valid_from
            if not _is_noop(fact, future_match):
                fact.superseded_by_id = future_match.old.id

        adds.append(fact)
        existing.append(_as_existing(fact))

    return adds, supersedes, noop_count


def _match_fact(fact: PreparedFact, old: ExistingFact) -> Match | None:
    similarity = _cosine(fact.embedding, old.embedding)
    entity_overlap = len(_entity_keys(fact.entities) & _entity_keys(old.entities))
    token_overlap = _jaccard(_tokens(fact.text), _tokens(old.text))
    entity_match = entity_overlap > 0 and (token_overlap >= 0.20 or entity_overlap >= 2)
    text_match = _canonical_text(fact.text) == _canonical_text(old.text)

    if text_match or similarity > VECTOR_SUPERSESSION_THRESHOLD or entity_match:
        return Match(
            old=old,
            similarity=max(similarity, 1.0 if text_match else 0.0),
            entity_overlap=entity_overlap,
            token_overlap=token_overlap,
        )
    return None


def _is_noop(fact: PreparedFact, match: Match) -> bool:
    return (
        _canonical_text(fact.text) == _canonical_text(match.old.text)
        or (
            fact.polarity == match.old.polarity
            and match.similarity >= 0.96
            and match.token_overlap >= 0.60
        )
    )


def _is_active_at(old: ExistingFact, timestamp: datetime) -> bool:
    starts_before = old.valid_from is None or old.valid_from <= timestamp
    ends_after = old.valid_to is None or old.valid_to > timestamp
    return starts_before and ends_after


def _as_existing(fact: PreparedFact) -> ExistingFact:
    return ExistingFact(
        id=fact.id,
        text=fact.text,
        polarity=fact.polarity,
        entities=fact.entities,
        embedding=fact.embedding,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
    )


async def _create_facts(
    tx: Any,
    user_id: str,
    session_id: str,
    timestamp: datetime,
    facts: list[PreparedFact],
) -> None:
    result = await tx.run(
        """
        MATCH (s:Session {user_id: $user_id, id: $session_id})
        UNWIND $facts AS f
          CREATE (fact:Fact {
            id: f.id,
            user_id: $user_id,
            text: f.text,
            embedding: f.embedding,
            polarity: f.polarity,
            confidence: f.confidence,
            created_at: datetime($timestamp),
            valid_from: datetime(f.valid_from)
          })
          SET fact.valid_to =
            CASE WHEN f.valid_to IS NULL THEN NULL ELSE datetime(f.valid_to) END
          MERGE (s)-[:CONTAINS_FACT]->(fact)
          FOREACH (entity IN f.entities |
            MERGE (ent:Entity {user_id: $user_id, name: entity.name})
              ON CREATE SET ent.type = entity.type
              ON MATCH SET ent.type = coalesce(ent.type, entity.type)
            MERGE (fact)-[:MENTIONS]->(ent)
          )
        """,
        user_id=user_id,
        session_id=session_id,
        timestamp=_iso_z(timestamp),
        facts=[_fact_params(fact) for fact in facts],
    )
    await result.consume()


async def _write_supersedes(
    tx: Any,
    user_id: str,
    timestamp: datetime,
    facts: list[PreparedFact],
) -> None:
    result = await tx.run(
        """
        UNWIND $facts AS f
          MATCH (newF:Fact {user_id: $user_id, id: f.id})
          MATCH (old:Fact {user_id: $user_id, id: f.supersedes_id})
          SET old.valid_to = datetime($timestamp)
          MERGE (newF)-[:SUPERSEDES]->(old)
        """,
        user_id=user_id,
        timestamp=_iso_z(timestamp),
        facts=[_fact_params(fact) for fact in facts],
    )
    await result.consume()


async def _write_future_supersedes(
    tx: Any, user_id: str, facts: list[PreparedFact]
) -> None:
    result = await tx.run(
        """
        UNWIND $facts AS f
          MATCH (newF:Fact {user_id: $user_id, id: f.id})
          MATCH (future:Fact {user_id: $user_id, id: f.superseded_by_id})
          MERGE (future)-[:SUPERSEDES]->(newF)
        """,
        user_id=user_id,
        facts=[_fact_params(fact) for fact in facts],
    )
    await result.consume()


def _fact_params(fact: PreparedFact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "text": fact.text,
        "polarity": fact.polarity,
        "confidence": fact.confidence,
        "entities": [
            {"name": entity.name, "type": entity.type or "unknown"}
            for entity in fact.entities
        ],
        "embedding": fact.embedding,
        "valid_from": _iso_z(fact.valid_from),
        "valid_to": _iso_z(fact.valid_to) if fact.valid_to is not None else None,
        "supersedes_id": fact.supersedes_id,
        "superseded_by_id": fact.superseded_by_id,
    }


def _validate_embedding(embedding: Any) -> list[float]:
    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"embedding must be a {EMBEDDING_DIMENSIONS}-dimension float list"
        )
    values = [float(value) for value in embedding]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("embedding contains non-finite values")
    return values


def _optional_embedding(embedding: Any) -> list[float] | None:
    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSIONS:
        return None
    try:
        return _validate_embedding(embedding)
    except (TypeError, ValueError):
        return None


def _cosine(left: list[float], right: list[float] | None) -> float:
    if right is None:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _entity_keys(entities: list[EntityCandidate]) -> set[str]:
    keys: set[str] = set()
    for entity in entities:
        key = _canonical_text(entity.name)
        if key and key not in GENERIC_ENTITY_NAMES:
            keys.add(key)
    return keys


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _canonical_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def _parse_timestamp(value: str) -> datetime:
    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_native"):
        parsed = value.to_native()
    else:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
