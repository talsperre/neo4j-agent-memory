"""One-example smoke test against LongMemEval_S.

Uses example 0 from the dataset. Verifies the full pipeline end-to-end:
Neo4j schema -> MCP cypher -> agent ingest loop -> embedding tool ->
agent retrieval loop -> answer -> judge.

Run with:
  PYTHONPATH=. uv run python -m eval.smoke
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
    """LongMemEval format: '2023/05/30 (Tue) 23:40' -> ISO-8601 UTC."""
    parts = date_str.replace("(", "").replace(")", "").split()
    date_part = parts[0]
    time_part = parts[2] if len(parts) > 2 else "00:00"
    dt = datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_sessions(example: dict) -> list[Session]:
    sessions = [
        Session(id=sid, timestamp=_parse_lme_date(sdate), messages=turns)
        for sid, sdate, turns in zip(
            example["haystack_session_ids"],
            example["haystack_dates"],
            example["haystack_sessions"],
        )
    ]
    sessions.sort(key=lambda s: s.timestamp)
    return sessions


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

    print(f"qid:       {example['question_id']}")
    print(f"type:      {example['question_type']}")
    print(f"question:  {example['question']}")
    print(f"answer:    {example['answer']}")
    print(f"date:      {example['question_date']}")
    print(f"sessions:  {len(example['haystack_sessions'])}")
    print()

    provider = Neo4jAgentProvider()
    await provider.initialize()
    client = anthropic.AsyncAnthropic()

    user_id = example["question_id"]
    question_date = _parse_lme_date(example["question_date"])

    try:
        await provider.clear(user_id)

        sessions = _build_sessions(example)
        t0 = time.time()
        print(f"[ingest] starting {len(sessions)} sessions...", flush=True)
        await provider.ingest(sessions, user_id)
        print(f"[ingest] done in {time.time() - t0:.1f}s")

        t0 = time.time()
        print(f"[search] starting...", flush=True)
        context = await provider.search(
            example["question"], user_id, as_of=question_date
        )
        print(f"[search] done in {time.time() - t0:.1f}s, {len(context)} facts")
        print(f"[search] context: {json.dumps(context, indent=2)[:800]}")

        t0 = time.time()
        hypothesis = await answer_question(
            client, example["question"], context, question_date
        )
        print(f"[answer] done in {time.time() - t0:.1f}s")
        print(f"[answer] hypothesis: {hypothesis}")

        correct = await judge(client, example["question"], example["answer"], hypothesis)
        print()
        print(f"VERDICT: {'CORRECT' if correct else 'WRONG'}")
        print(f"  ground truth: {example['answer']}")
        print(f"  hypothesis:   {hypothesis}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
