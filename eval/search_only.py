"""Run search + answer + judge against existing Neo4j data (no ingest).

Uses example 0 from LongMemEval but skips clearing/ingesting -- works with
whatever facts are already in the graph for that user_id.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import anthropic

from neo4j_memory.prompts import ANSWER_PROMPT
from neo4j_memory.provider import Neo4jAgentProvider

DATA_PATH = Path("data/longmemeval_full.json")
MODEL = "claude-opus-4-7"


def _parse_lme_date(date_str: str) -> str:
    parts = date_str.replace("(", "").replace(")", "").split()
    date_part = parts[0]
    time_part = parts[2] if len(parts) > 2 else "00:00"
    dt = datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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

    print(f"qid:        {user_id}")
    print(f"question:   {question}")
    print(f"ground truth: {example['answer']}")
    print(f"as_of:      {as_of}")
    print()

    provider = Neo4jAgentProvider()
    client = anthropic.AsyncAnthropic()
    try:
        print("[search] starting...", flush=True)
        context = await provider.search(question, user_id, as_of=as_of)
        print(f"[search] returned {len(context)} facts")
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
