"""Search-based memory eval on ONE LongMemEval row.

Storage:    each session is dumped verbatim as ./sessions/YYYY-MM-DD.md (no LLM).
Retrieval:  Claude has Bash/Grep/Read/Glob. It greps the sessions/ dir to
            find relevant context and answers. Hard 10s wall-clock budget.
Judge:      separate claude call with no tools, strict prompt.

Run:
  python eval_one.py
"""

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


DATASET_PATH = "/tmp/lme_oracle.json"
TARGET_QID = "9aaed6a3"

MODEL_QUERY = "claude-sonnet-4-5"
MODEL_JUDGE = "claude-sonnet-4-5"

QUERY_BUDGET_SECONDS = 10
HARD_TIMEOUT_SECONDS = 120
QUERY_EFFORT = None  # claude --effort: low / medium / high / xhigh / max (None = default)


QUERY_SYSTEM = """You answer questions about the user's past conversations, which are stored as plain-text files in ./sessions/, one per session, named by date.

Use the available shell and file-reading tools to find relevant context. Answer based on what is in the stored conversations. Do not invent information that is not there.

Distinguish factual questions ("what happened", "how many", "when did I", "what is X") from recommendation questions ("suggest X", "recommend X", "any advice on X", "what X should I try").

- For factual questions: answer only from what is explicitly in the conversations. Distinguish "the conversations state X" from "the conversations don't mention X" — if a topic is absent, say it's absent rather than translating absence into a definitive zero, "no", or "none". If the question's premise (a specific role, event, person, or thing) is not in the conversations, do not substitute a similar-but-different one; state that the specific thing asked about is not mentioned.
- For recommendation questions: the user is asking you to apply their known preferences to a new situation. Use their stated preferences from past conversations to inform the suggestion, even if the specific topic of the question (e.g., a city, a recipe, a tool) was not previously discussed. The same person likely has the same preferences across related contexts.

End your reply with a line of exactly: `ANSWER: <your answer>`
"""


JUDGE_SYSTEM = """You evaluate a response against a gold answer for a memory benchmark.

Rules:
- CORRECT = response contains the correct answer, is semantically equivalent, or shows the right intermediate steps.
- INCORRECT = response only contains a subset of the required information, contradicts the gold, or hallucinates.
- For TEMPORAL questions: off-by-one day on a date or duration is acceptable.
- For ABSTENTION questions: CORRECT only if the system declined to answer / said the info was not provided.
- No partial credit for lists.

Return ONLY valid JSON in this exact format (no prose, no code fences):
{"score": 0 or 1, "label": "correct" or "incorrect", "explanation": "one sentence"}
"""


def format_session_file(date_header: str, turns: list[dict]) -> str:
    lines = [f"# Session {date_header}", ""]
    for t in turns:
        lines.append(f"{t['role'].upper()}: {t['content']}")
        lines.append("")
    return "\n".join(lines)


def write_sessions_verbatim(workdir: Path, example: dict) -> None:
    sessions_dir = workdir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    for date_str, turns in zip(example["haystack_dates"], example["haystack_sessions"]):
        # Filename: just the date (YYYY-MM-DD)
        m = re.match(r"(\d{4})/(\d{2})/(\d{2})", date_str)
        if m:
            fname = f"{m.group(1)}-{m.group(2)}-{m.group(3)}.md"
        else:
            fname = f"unknown-{abs(hash(date_str)) % 10000}.md"
        path = sessions_dir / fname
        # If two sessions land on the same date, append a suffix
        i = 1
        while path.exists():
            path = sessions_dir / f"{path.stem}_{i}{path.suffix}"
            i += 1
        path.write_text(format_session_file(date_str, turns))


def run_claude(
    user_prompt: str,
    system_prompt: str,
    model: str,
    cwd: Path,
    tools: str,
    effort: str | None = None,
    timeout: int = HARD_TIMEOUT_SECONDS,
) -> dict:
    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
        "--system-prompt", system_prompt,
        "--tools", tools,
        "--disallowedTools", "Edit Write Task WebFetch WebSearch TaskCreate TaskUpdate TaskList Skill",
        "--max-budget-usd", "1",
    ]
    if effort:
        cmd += ["--effort", effort]
    cmd.append(user_prompt)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}\nSTDERR:\n{proc.stderr[:2000]}")
    return json.loads(proc.stdout)


def extract_response_text(result: dict) -> str:
    if isinstance(result, dict):
        if "result" in result and isinstance(result["result"], str):
            return result["result"]
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                return block.get("text", "")
    return json.dumps(result)[:500]


def extract_answer(response: str) -> str:
    m = re.search(r"ANSWER:\s*(.+)", response, flags=re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def main():
    data = json.load(open(DATASET_PATH))
    example = next((d for d in data if d["question_id"] == TARGET_QID), None)
    if not example:
        raise SystemExit(f"qid {TARGET_QID} not found in {DATASET_PATH}")

    print(f"qid:        {example['question_id']}")
    print(f"type:       {example['question_type']}")
    print(f"q_date:     {example['question_date']}")
    print(f"question:   {example['question']}")
    print(f"gold:       {example['answer']}")
    print(f"sessions:   {len(example['haystack_sessions'])}")

    workdir = Path(tempfile.mkdtemp(prefix="lme_eval_"))
    print(f"workdir:    {workdir}\n")

    # ── Ingestion: just write files verbatim, no LLM call ────────────────────
    t0 = time.monotonic()
    write_sessions_verbatim(workdir, example)
    ingest_secs = time.monotonic() - t0
    print(f"=== INGESTION  ({ingest_secs*1000:.0f}ms) ===")
    for p in sorted((workdir / "sessions").glob("*.md")):
        print(f"  {p.relative_to(workdir)}  ({p.stat().st_size} bytes)")

    # ── Query: agent greps sessions ──────────────────────────────────────────
    print(f"\n=== QUERY (budget: {QUERY_BUDGET_SECONDS}s) ===")
    query_prompt = (
        f"Reference date (today): {example['question_date']}\n\n"
        f"Question: {example['question']}"
    )
    t0 = time.monotonic()
    result = run_claude(
        user_prompt=query_prompt,
        system_prompt=QUERY_SYSTEM,
        model=MODEL_QUERY,
        cwd=workdir,
        tools="Bash Read Grep Glob",
        effort=QUERY_EFFORT,
    )
    query_secs = time.monotonic() - t0

    response = extract_response_text(result)
    qcost = result.get("total_cost_usd", 0)
    qturns = result.get("num_turns", "?")

    budget_status = "✓ within budget" if query_secs <= QUERY_BUDGET_SECONDS else "✗ OVER BUDGET"
    print(f"({query_secs:.1f}s, {qturns} turns, ${qcost:.4f})  [{budget_status}]")
    print(response)

    answer = extract_answer(response)

    # ── Judge ────────────────────────────────────────────────────────────────
    print("\n=== JUDGE ===")
    judge_prompt = (
        f"Question: {example['question']}\n"
        f"Gold answer: {example['answer']}\n"
        f"Response: {answer}"
    )
    t0 = time.monotonic()
    result = run_claude(
        user_prompt=judge_prompt,
        system_prompt=JUDGE_SYSTEM,
        model=MODEL_JUDGE,
        cwd=workdir,
        tools="",
    )
    judge_secs = time.monotonic() - t0

    raw_judge = extract_response_text(result).strip()
    if raw_judge.startswith("```"):
        raw_judge = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_judge, flags=re.DOTALL).strip()
    try:
        verdict = json.loads(raw_judge)
    except json.JSONDecodeError:
        verdict = {"score": 0, "label": "parse_error", "explanation": raw_judge[:300]}

    print(f"({judge_secs:.1f}s, ${result.get('total_cost_usd', 0):.4f})")
    print(f"score:       {verdict.get('score')}  ({verdict.get('label')})")
    print(f"explanation: {verdict.get('explanation','')}")

    print(f"\nworkdir kept at: {workdir}")


if __name__ == "__main__":
    main()
