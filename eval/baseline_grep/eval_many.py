"""Run the search-based eval on N=10 diverse LongMemEval examples in parallel.

Same architecture as eval_one.py: verbatim session files + Claude with Bash/grep.
Picks examples across all question types for a balanced sample.
"""

import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Optional CLI: python eval_many.py <type_filter>
TYPE_FILTER = sys.argv[1] if len(sys.argv) > 1 else None


DATASET_PATH = "/tmp/lme_oracle.json"

MODEL_QUERY = "claude-sonnet-4-5"
MODEL_JUDGE = "claude-sonnet-4-5"

HARD_TIMEOUT_SECONDS = 180
MAX_WORKERS = 10

# How many examples per question_type (caps total)
PER_TYPE_LIMIT = 9999
TOTAL_EXAMPLES = 9999


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
        m = re.match(r"(\d{4})/(\d{2})/(\d{2})", date_str)
        fname = f"{m.group(1)}-{m.group(2)}-{m.group(3)}.md" if m else f"unk-{abs(hash(date_str)) % 10000}.md"
        path = sessions_dir / fname
        i = 1
        while path.exists():
            path = sessions_dir / f"{path.stem}_{i}{path.suffix}"
            i += 1
        path.write_text(format_session_file(date_str, turns))


def run_claude(user_prompt, system_prompt, model, cwd, tools, timeout=HARD_TIMEOUT_SECONDS):
    cmd = [
        "claude",
        "--print", "--output-format", "json",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
        "--system-prompt", system_prompt,
        "--tools", tools,
        "--disallowedTools", "Edit Write Task WebFetch WebSearch TaskCreate TaskUpdate TaskList Skill",
        "--max-budget-usd", "1",
        user_prompt,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def extract_response_text(result: dict) -> str:
    if isinstance(result, dict):
        if "result" in result and isinstance(result["result"], str):
            return result["result"]
    return json.dumps(result)[:500]


def extract_answer(response: str) -> str:
    m = re.search(r"ANSWER:\s*(.+)", response, flags=re.DOTALL)
    return m.group(1).strip() if m else response.strip()


def evaluate_one(example: dict) -> dict:
    """Full pipeline on a single example. Returns a result dict (never raises)."""
    qid = example["question_id"]
    out: dict = {
        "qid": qid,
        "type": example["question_type"],
        "question": example["question"],
        "gold": example["answer"],
    }
    workdir = Path(tempfile.mkdtemp(prefix=f"lme_{qid}_"))
    try:
        write_sessions_verbatim(workdir, example)

        # Query
        t0 = time.monotonic()
        query_prompt = (
            f"Reference date (today): {example['question_date']}\n\n"
            f"Question: {example['question']}"
        )
        qres = run_claude(query_prompt, QUERY_SYSTEM, MODEL_QUERY, workdir, "Bash Read Grep Glob")
        out["query_secs"] = time.monotonic() - t0
        out["query_cost"] = qres.get("total_cost_usd", 0)
        out["query_turns"] = qres.get("num_turns", 0)
        response = extract_response_text(qres)
        out["response"] = response
        answer = extract_answer(response)
        out["answer"] = answer

        # Judge
        t0 = time.monotonic()
        judge_prompt = (
            f"Question: {example['question']}\n"
            f"Gold answer: {example['answer']}\n"
            f"Response: {answer}"
        )
        jres = run_claude(judge_prompt, JUDGE_SYSTEM, MODEL_JUDGE, workdir, "")
        out["judge_secs"] = time.monotonic() - t0
        out["judge_cost"] = jres.get("total_cost_usd", 0)
        raw_judge = extract_response_text(jres).strip()
        if raw_judge.startswith("```"):
            raw_judge = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_judge, flags=re.DOTALL).strip()
        try:
            verdict = json.loads(raw_judge)
        except json.JSONDecodeError:
            verdict = {"score": 0, "label": "parse_error", "explanation": raw_judge[:200]}
        out["score"] = verdict.get("score", 0)
        out["label"] = verdict.get("label", "?")
        out["judge_explanation"] = verdict.get("explanation", "")
    except subprocess.TimeoutExpired:
        out["error"] = "timeout"
        out["score"] = 0
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["score"] = 0
    return out


def pick_examples(dataset: list[dict]) -> list[dict]:
    """Pick TOTAL_EXAMPLES with up to PER_TYPE_LIMIT from each question_type, plus abstention variants.
    If TYPE_FILTER is set, restrict to examples whose question_type equals that filter."""
    filtered = dataset
    if TYPE_FILTER:
        filtered = [d for d in dataset if d.get("question_type") == TYPE_FILTER]

    by_type: dict[str, list[dict]] = defaultdict(list)
    for d in filtered:
        is_abs = "_abs" in d.get("question_id", "")
        key = d["question_type"] + ("_abs" if is_abs else "")
        by_type[key].append(d)

    picked = []
    for key in sorted(by_type):
        picked.extend(by_type[key][:PER_TYPE_LIMIT])
        if len(picked) >= TOTAL_EXAMPLES:
            break
    return picked[:TOTAL_EXAMPLES]


def main():
    data = json.load(open(DATASET_PATH))
    selected = pick_examples(data)

    print(f"Running {len(selected)} examples in parallel (max {MAX_WORKERS} concurrent)\n")
    for s in selected:
        marker = " (abstention)" if "_abs" in s["question_id"] else ""
        print(f"  - {s['question_id']}  [{s['question_type']}{marker}]  {s['question'][:70]}")
    print()

    t0 = time.monotonic()
    results = []
    suffix_inc = f"_{TYPE_FILTER}" if TYPE_FILTER else ""
    incremental_path = Path(f"/tmp/eval_many_results{suffix_inc}.json")
    progress_path = Path(f"/tmp/eval_many_progress{suffix_inc}.txt")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_one, ex): ex for ex in selected}
        for i, fut in enumerate(as_completed(futures), 1):
            ex = futures[fut]
            res = fut.result()
            results.append(res)
            mark = "✓" if res.get("score") else "✗"
            err = f"  ERR={res.get('error')}" if res.get("error") else ""
            secs = res.get("query_secs", 0) + res.get("judge_secs", 0)
            elapsed = time.monotonic() - t0
            line = f"  [{mark}] [{i:3d}/{len(selected)}]  {ex['question_id']:20s}  {res.get('type','?'):28s}  {secs:5.1f}s  (wall {elapsed:.0f}s){err}"
            print(line, flush=True)
            # Incremental save every example
            incremental_path.write_text(json.dumps(results, indent=2, default=str))
            correct_so_far = sum(1 for r in results if r.get("score"))
            progress_path.write_text(f"{i}/{len(selected)} done, {correct_so_far} correct ({correct_so_far/i*100:.1f}%), wall {elapsed:.0f}s\n")
    wall = time.monotonic() - t0

    # Sort results by qid for stable output
    results.sort(key=lambda r: r["qid"])

    # Persist to disk so we never re-run for analysis
    suffix = f"_{TYPE_FILTER}" if TYPE_FILTER else ""
    results_path = Path(f"/tmp/eval_many_results{suffix}.json")
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    total = len(results)
    correct = sum(1 for r in results if r.get("score"))
    print(f"Overall: {correct}/{total}  ({correct/total*100:.0f}%)")
    print(f"Wall clock: {wall:.1f}s")
    total_cost = sum(r.get("query_cost", 0) + r.get("judge_cost", 0) for r in results)
    print(f"Total cost: ${total_cost:.2f}")

    # Per-type breakdown
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        key = r.get("type", "?")
        if "_abs" in r["qid"]:
            key += " (abs)"
        by_type[key].append(r)
    print("\nBy question type:")
    for t, rs in sorted(by_type.items()):
        c = sum(1 for r in rs if r.get("score"))
        print(f"  {t:36s}  {c}/{len(rs)}")

    print("\nPer-example detail:")
    for r in results:
        mark = "✓" if r.get("score") else "✗"
        print(f"\n  [{mark}] {r['qid']}  ({r.get('type','?')})")
        print(f"      Q:    {r['question']}")
        print(f"      Gold: {str(r['gold'])[:200]}")
        print(f"      Got:  {str(r.get('answer','(none)'))[:300]}")
        if r.get("error"):
            print(f"      ERR:  {r['error']}")
        else:
            print(f"      Time: {r.get('query_secs',0):.1f}s query + {r.get('judge_secs',0):.1f}s judge   Cost: ${r.get('query_cost',0)+r.get('judge_cost',0):.3f}")
            print(f"      Judge: {r.get('judge_explanation','')}")


if __name__ == "__main__":
    main()
