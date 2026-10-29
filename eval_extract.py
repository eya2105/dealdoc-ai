"""Evaluate /extract on a small hand-checked set (eval/golden_extract.json).

For every field we check two things:
  correct  - the extracted value contains every expected string
  grounded - the chunks it cites really contain those strings (the citation is not made up)

    python eval_extract.py --retrieval-only      # no extraction LLM calls: only checks the retrieved evidence
    python eval_extract.py --retrieval-only --depth 20   # ...and at which rank the answer chunk first appears
    python eval_extract.py                       # one run
    python eval_extract.py --runs 3              # the LLM is not deterministic: count passes over N runs
    python eval_extract.py --runs 3 --diagnose   # for every field that ever failed: which stage, which chunks, what the LLM said

Needs the index (python ingest.py) and the LLM key. Free Groq tiers cap tokens per minute (8000 for
gpt-oss-120b): one run is ~8 small LLM calls and may take a couple of minutes while the client waits.
"""
import argparse
import json
import time
from pathlib import Path

import config
from eval import normalize


# ── Scoring (pure, unit-tested) ─────────────────────────────────────────

def score_field(expected: list[str], value, source_texts: list[str]) -> tuple[bool, bool]:
    """Return (correct, grounded) for one field."""
    if value is None:
        return False, False
    flat = normalize(json.dumps(value, ensure_ascii=False))
    correct = all(normalize(e) in flat for e in expected)
    grounded = correct and all(
        any(normalize(e) in normalize(t) for t in source_texts) for e in expected
    )
    return correct, grounded


def evidence_has_answer(expected: list[str], texts: list[str]) -> bool:
    """Did retrieval put every expected string somewhere in this field's evidence chunks?"""
    return all(any(normalize(e) in normalize(t) for t in texts) for e in expected)


def first_answer_rank(expected: list[str], chunks: list[dict]) -> int | None:
    """1-based rank of the first retrieved chunk that contains every expected string, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if evidence_has_answer(expected, [chunk["text"]]):
            return rank
    return None


def classify(correct: bool, retrieved: bool) -> str:
    """Which stage failed? A wrong value with the answer in the evidence is the LLM's fault."""
    if correct:
        return "ok"
    return "LLM miss (answer WAS in the evidence)" if retrieved else "retrieval miss (answer NOT in the evidence)"


def stage_of(passes: int, runs: int, retrieved: bool) -> str:
    """Table label: ok / flaky (sometimes right) / LLM / retrieval."""
    if passes == runs:
        return "ok"
    if passes > 0:
        return "flaky"
    return "LLM" if retrieved else "retrieval"


# ── Run ─────────────────────────────────────────────────────────────────

def _container():
    """The wired application (tests replace this function with a fake)."""
    from container import get_container

    return get_container()


def print_chunks(evidence: list[dict], expected: list[str]) -> None:
    for c in evidence:
        hit = any(normalize(e) in normalize(c["text"]) for e in expected)
        chunk_no = c["chunk_id"].split("__")[-1]
        rr = f" rr={c['rerank_score']:.1f}" if "rerank_score" in c else ""
        print(f"    {'*' if hit else ' '} {c['metadata']['doc_id']}/{chunk_no}{rr}: {c['text'][:110]}")


def retrieval_only(golden: dict, depth: int, extractor) -> None:
    """Check ONLY the retrieval stage (no extraction LLM calls, so no token quota burned).

    `depth` chunks are retrieved per field so we can see at which rank the answer first appears;
    the extraction itself only uses the first EXTRACT_CHUNKS_PER_FIELD of them.
    """
    used = config.EXTRACT_CHUNKS_PER_FIELD
    expected = golden["expected"]
    _, queries, evidence = extractor.retrieve_evidence(golden["deal_id"], list(expected), per_field=depth)  # may make ONE small LLM call to write queries
    ranks = {f: first_answer_rank(exp, evidence[f]) for f, exp in expected.items()}
    for field, exp in expected.items():
        rank = ranks[field]
        where = f"answer chunk first at rank {rank}" if rank else f"answer NOT in the top {depth}"
        print(f"\n{field}: {where}   expected {exp}")
        print(f"  query: {queries[field]}")
        print_chunks(evidence[field][:max(used, 3)], exp)
    in_used = sum(r is not None and r <= used for r in ranks.values())
    in_depth = sum(r is not None for r in ranks.values())
    print(f"\nanswer within the {used} chunks the LLM sees: {in_used}/{len(expected)} fields"
          f"   within the top {depth}: {in_depth}/{len(expected)}   (* = chunk contains an expected string)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=1, help="repeat the LLM step N times (it varies between runs)")
    p.add_argument("--depth", type=int, default=config.EXTRACT_CHUNKS_PER_FIELD,
                   help="with --retrieval-only: chunks retrieved per field (max 20), to see at which rank the answer appears")
    p.add_argument("--retrieval-only", action="store_true", help="only check retrieval (no extraction LLM calls)")
    p.add_argument("--diagnose", action="store_true", help="show retrieved chunks and the raw LLM answer for failing fields")
    args = p.parse_args()

    from extraction import ExtractionError

    container = _container()
    golden = json.loads(Path(config.GOLDEN_EXTRACT_PATH).read_text(encoding="utf-8"))
    if args.retrieval_only:
        retrieval_only(golden, min(max(args.depth, 1), config.MAX_TOP_K), container.extractor)
        return

    expected = golden["expected"]
    runs = []
    for i in range(1, args.runs + 1):
        started = time.time()
        try:
            runs.append(container.extractor.run(golden["deal_id"], list(expected)))
        except ExtractionError as err:
            raise SystemExit(f"\nRun {i} failed: {err}\n"
                             "Likely a rate limit (tokens per minute). Try --retrieval-only, fewer --runs, "
                             "EXTRACT_CHUNKS_PER_FIELD=2 in config.py, or a higher Groq tier.")
        print(f"run {i}/{args.runs} took {time.time() - started:.0f}s")

    def cited_texts(res: dict) -> list[str]:
        return container.store.get_texts([s["chunk_id"] for s in res["sources"]])

    n = args.runs
    print(f"\n{'field':<28}{'correct':<9}{'grounded':<10}{'stage':<11}last value")
    totals = {"correct": 0, "grounded": 0}
    failing = []
    for field, exp in expected.items():
        scores = [score_field(exp, r.results[field]["value"], cited_texts(r.results[field])) for r in runs]
        passes = sum(c for c, _ in scores)
        grounded = sum(g for _, g in scores)
        # retrieval does not depend on the LLM, so any run's evidence tells the same story
        retrieved = evidence_has_answer(exp, [c["text"] for c in runs[-1].evidence[field]])
        stage = stage_of(passes, n, retrieved)
        totals["correct"] += passes
        totals["grounded"] += grounded
        if stage != "ok":
            failing.append((field, exp, passes, retrieved))
        shown = json.dumps(runs[-1].results[field]["value"], ensure_ascii=False)
        print(f"{field:<28}{f'{passes}/{n}':<9}{f'{grounded}/{n}':<10}{stage:<11}{shown[:55]}")

    total = len(expected) * n
    in_evidence = sum(evidence_has_answer(exp, [c["text"] for c in runs[-1].evidence[f]]) for f, exp in expected.items())
    prompt_chars = sum(len(c) for c in runs[-1].contexts.values())
    print(f"\ncorrect {totals['correct']}/{total}   grounded {totals['grounded']}/{total}   "
          f"(retrieval: answer in the evidence for {in_evidence}/{len(expected)} fields; prompts: {prompt_chars} chars total)")

    if not args.diagnose or not failing:
        return

    last = runs[-1]
    print("\n" + "=" * 78)
    for field, exp, passes, retrieved in failing:
        print(f"\n{field}: {classify(False, retrieved)}   [correct in {passes}/{n} runs]")
        print(f"  expected : {exp}")
        print(f"  query    : {last.queries[field]}")
        print_chunks(last.evidence[field], exp)
        for i, r in enumerate(runs, start=1):
            print(f"  LLM run {i}: {json.dumps(r.raw[field], ensure_ascii=False)[:200]}")
    print("\n(* = chunk contains an expected string)")


if __name__ == "__main__":
    main()
