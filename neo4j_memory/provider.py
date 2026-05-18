import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import anthropic
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from .agent import neo4j_mcp_session, run_agent_loop
from .extract import DEFAULT_INGEST_MODEL, extract_facts
from .prompts import RETRIEVE_SYSTEM_PROMPT
from .schema import ensure_schema
from .writer import write_session_facts

INGEST_CONCURRENCY = 4
INGEST_MODEL = DEFAULT_INGEST_MODEL
SEARCH_MODEL = "claude-opus-4-7"
SEARCH_EFFORT = "xhigh"

JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.S)


@dataclass
class Session:
    id: str
    timestamp: str
    messages: list[dict]


class Neo4jAgentProvider:
    name = "neo4j-agent"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic()
        self.openai = AsyncOpenAI()
        self.driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )

    async def initialize(self):
        await ensure_schema(self.driver)

    async def close(self):
        await self.driver.close()
        await self.openai.close()

    async def ingest(self, sessions: list[Session], user_id: str) -> list[str]:
        sem = asyncio.Semaphore(INGEST_CONCURRENCY)

        async def _one(session: Session) -> str:
            async with sem:
                normalized = Session(
                    id=session.id,
                    timestamp=_normalize_timestamp(session.timestamp),
                    messages=session.messages,
                )
                facts = await extract_facts(
                    self.client,
                    normalized,
                    model=INGEST_MODEL,
                )
                await write_session_facts(
                    self.driver,
                    self.openai,
                    user_id=user_id,
                    session=normalized,
                    facts=facts,
                )
                return session.id

        results = await asyncio.gather(
            *(_one(s) for s in sessions), return_exceptions=True
        )
        succeeded = [r for r in results if isinstance(r, str)]
        failed = [
            (s.id, r) for s, r in zip(sessions, results) if isinstance(r, BaseException)
        ]
        for sid, exc in failed:
            print(f"[ingest] session {sid} failed: {type(exc).__name__}: {exc}", flush=True)
        return succeeded

    async def search(self, question: str, user_id: str, as_of: str) -> list[dict]:
        payload = {
            "operation": "retrieve_memory",
            "user_id": user_id,
            "as_of": _normalize_timestamp(as_of),
            "question": question,
        }
        user_text = "RETRIEVE MEMORY:\n" + json.dumps(payload)
        async with neo4j_mcp_session() as mcp:
            text = await run_agent_loop(
                self.client,
                self.openai,
                mcp,
                RETRIEVE_SYSTEM_PROMPT,
                user_text,
                required_user_id=user_id,
                max_iterations=12,
                model=SEARCH_MODEL,
                effort=SEARCH_EFFORT,
            )
        return _extract_json_array(text)

    async def clear(self, user_id: str):
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (u:User {id: $id})
                OPTIONAL MATCH (u)-[:HAS_SESSION]->(session:Session)
                OPTIONAL MATCH (session)-[:CONTAINS_FACT]->(fact:Fact {user_id: $id})
                OPTIONAL MATCH (fact)-[:MENTIONS]->(entity:Entity {user_id: $id})
                DETACH DELETE entity, fact, session, u
                """,
                id=user_id,
            )


def _normalize_timestamp(value: str) -> str:
    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"

    parsed = datetime.fromisoformat(timestamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_json_array(text: str) -> list[dict]:
    match = JSON_BLOCK_RE.search(text)
    if match is None:
        raise ValueError("retrieval agent did not return a fenced JSON block")
    return json.loads(match.group(1))
