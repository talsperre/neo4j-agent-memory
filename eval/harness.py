"""LongMemEval_S harness with parallel sessions and parallel examples.

Skips examples whose user already has a non-zero session count in Neo4j
(so it acts as a resume on partial runs). Per-example results land in
results.jsonl as they complete; final accuracy printed at the end.

Usage:
  PYTHONPATH=. uv run python -m eval.harness                   # all 500
  LIMIT=10 PYTHONPATH=. uv run python -m eval.harness          # first 10
  EXAMPLE_CONCURRENCY=2 LIMIT=10 ... python -m eval.harness    # tune
"""

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import anthropic

from neo4j_memory.prompts import ANSWER_PROMPT
from neo4j_memory.provider import Neo4jAgentProvider, Session

DATA_PATH = Path(os.environ.get("LONGMEMEVAL_PATH", "data/longmemeval_full.json"))
RESULTS_PATH = Path(os.environ.get("RESULTS_PATH", "results.jsonl"))
MODEL = "claude-opus-4-7"
JUDGE_MODEL = "claude-haiku-4-5"

EXAMPLE_CONCURRENCY = int(os.environ.get("EXAMPLE_CONCURRENCY", "2"))
LIMIT = int(os.environ.get("LIMIT", "0")) or None


def _parse_lme_date(date_str: str) -> str:
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


async def _already_done_users(provider: Neo4jAgentProvider) -> set[str]:
    """Users with at least one Session node -- treated as already ingested."""
    async with provider.driver.session() as s:
        result = await s.run(
            "MATCH (u:User)-[:HAS_SESSION]->(:Session) RETURN DISTINCT u.id AS id"
        )
        return {record["id"] async for record in result}


async def _already_done_results() -> set[str]:
    """user_ids already present in results.jsonl."""
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    for line in RESULTS_PATH.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["qid"])
    return done


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
        model=JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip().upper().startswith("YES")


async def run_one(
    provider: Neo4jAgentProvider,
    client: anthropic.AsyncAnthropic,
    example: dict,
    already_ingested: bool,
) -> dict:
    user_id = example["question_id"]
    question = example["question"]
    as_of = _parse_lme_date(example["question_date"])
    t0 = time.time()

    if not already_ingested:
        sessions = _build_sessions(example)
        await provider.ingest(sessions, user_id)

    context = await provider.search(question, user_id, as_of=as_of)
    hypothesis = await answer_question(client, question, context, as_of)
    correct = await judge(client, question, example["answer"], hypothesis)

    return {
        "qid": user_id,
        "question_type": example.get("question_type"),
        "question": question,
        "ground_truth": example["answer"],
        "hypothesis": hypothesis,
        "correct": correct,
        "elapsed_s": time.time() - t0,
        "context_size": len(context),
    }


async def main():
    examples = json.loads(DATA_PATH.read_text())
    if LIMIT:
        examples = examples[:LIMIT]
    print(f"loaded {len(examples)} examples (LIMIT={LIMIT}, concurrency={EXAMPLE_CONCURRENCY})")

    provider = Neo4jAgentProvider()
    await provider.initialize()
    client = anthropic.AsyncAnthropic()

    try:
        ingested_users = await _already_done_users(provider)
        scored_users = await _already_done_results()
        print(f"already ingested in Neo4j: {len(ingested_users)} users")
        print(f"already scored in {RESULTS_PATH}: {len(scored_users)} users")

        to_run = [ex for ex in examples if ex["question_id"] not in scored_users]
        print(f"will run: {len(to_run)} examples")
        if not to_run:
            return

        sem = asyncio.Semaphore(EXAMPLE_CONCURRENCY)
        completed = 0
        correct_count = 0
        lock = asyncio.Lock()

        async def bounded(ex):
            nonlocal completed, correct_count
            async with sem:
                try:
                    r = await run_one(
                        provider, client, ex,
                        already_ingested=ex["question_id"] in ingested_users,
                    )
                except Exception as exc:
                    r = {
                        "qid": ex["question_id"],
                        "question_type": ex.get("question_type"),
                        "question": ex["question"],
                        "ground_truth": ex["answer"],
                        "hypothesis": None,
                        "correct": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                async with lock:
                    with RESULTS_PATH.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    completed += 1
                    correct_count += int(r["correct"])
                    print(
                        f"[{completed}/{len(to_run)}] {r['qid']} "
                        f"{'OK' if r.get('correct') else 'WRONG'} "
                        f"running={correct_count}/{completed}",
                        flush=True,
                    )
                return r

        await asyncio.gather(*(bounded(ex) for ex in to_run))

        total = sum(1 for _ in RESULTS_PATH.read_text().splitlines() if _.strip())
        correct_total = 0
        for line in RESULTS_PATH.read_text().splitlines():
            if line.strip() and json.loads(line).get("correct"):
                correct_total += 1
        print()
        print(f"FINAL accuracy: {correct_total}/{total} = {correct_total/total:.3f}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
