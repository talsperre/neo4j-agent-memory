"""Resume ingest: only process sessions not already in Neo4j, then search/answer/judge.

Run with:
  PYTHONPATH=. uv run python -m eval.resume
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import anthropic

from neo4j_memory.prompts import ANSWER_PROMPT
from neo4j_memory.provider import Neo4jAgentProvider, Session

DATA_PATH = Path("data/longmemeval_full.json")
MODEL = "claude-opus-4-7"


def _parse_lme_date(date_str: str) -> str:
    parts = date_str.replace("(", "").replace(")", "").split()
    date_part = parts[0]
    time_part = parts[2] if len(parts) > 2 else "00:00"
    dt = datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _already_ingested_session_ids(provider: Neo4jAgentProvider, user_id: str) -> set[str]:
    async with provider.driver.session() as s:
        result = await s.run(
            "MATCH (u:User {id: $id})-[:HAS_SESSION]->(session:Session) "
            "RETURN session.id AS id",
            id=user_id,
        )
        return {record["id"] async for record in result}


async def answer_question(client, question, context, as_of) -> str:
    prompt = ANSWER_PROMPT.format(
        question=question, context=json.dumps(context), as_of=as_of
    )
    async with client.messages.stream(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = await stream.get_final_message()
    return "".join(b.text for b in message.content if b.type == "text").strip()


async def judge(client, question, ground_truth, hypothesis) -> bool:
    prompt = (
        f"Question: {question}\nGround truth: {ground_truth}\n"
        f"Candidate answer: {hypothesis}\n\n"
        "Does the candidate answer convey the same information as ground truth? "
        "Answer YES or NO."
    )
    message = await client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip().upper().startswith("YES")


async def main():
    examples = json.loads(DATA_PATH.read_text())
    example = examples[0]
    user_id = example["question_id"]
    question = example["question"]
    as_of = _parse_lme_date(example["question_date"])

    print(f"qid:          {user_id}")
    print(f"question:     {question}")
    print(f"ground truth: {example['answer']}")
    print(f"as_of:        {as_of}")
    print()

    provider = Neo4jAgentProvider()
    await provider.initialize()
    client = anthropic.AsyncAnthropic()

    try:
        done = await _already_ingested_session_ids(provider, user_id)
        print(f"already ingested: {len(done)} sessions")

        all_sessions = [
            Session(id=sid, timestamp=_parse_lme_date(sdate), messages=turns)
            for sid, sdate, turns in zip(
                example["haystack_session_ids"],
                example["haystack_dates"],
                example["haystack_sessions"],
            )
            if sid not in done
        ]
        all_sessions.sort(key=lambda s: s.timestamp)
        print(f"to ingest:        {len(all_sessions)} sessions")
        print()

        if all_sessions:
            t0 = time.time()
            failed = []
            for i, s in enumerate(all_sessions, 1):
                ts0 = time.time()
                try:
                    await provider.ingest([s], user_id)
                    print(
                        f"[ingest {i}/{len(all_sessions)}] {s.id} done in "
                        f"{time.time() - ts0:.1f}s (total {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    failed.append(s.id)
                    print(
                        f"[ingest {i}/{len(all_sessions)}] {s.id} FAILED in "
                        f"{time.time() - ts0:.1f}s: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
            if failed:
                print(f"[ingest] {len(failed)} sessions failed: {failed}")

        t0 = time.time()
        print(f"[search] starting...", flush=True)
        context = await provider.search(question, user_id, as_of=as_of)
        print(f"[search] done in {time.time() - t0:.1f}s, {len(context)} facts")
        print("[search] context:")
        print(json.dumps(context, indent=2))
        print()

        hypothesis = await answer_question(client, question, context, as_of)
        print(f"[answer] hypothesis: {hypothesis}")
        print()

        correct = await judge(client, question, example["answer"], hypothesis)
        print(f"VERDICT: {'CORRECT' if correct else 'WRONG'}")
        print(f"  expected: {example['answer']}")
        print(f"  got:      {hypothesis}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
