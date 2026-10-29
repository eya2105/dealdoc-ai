"""Retrieval evaluation: recall@k and MRR on a hand-written golden set.

A question counts as a HIT at rank r if the r-th retrieved chunk contains the
expected answer string (and comes from `expected_doc`, when the item sets one).

    python eval.py --check-golden          # no model / no API key: are answers inside single chunks?
    python eval.py                         # k=5, vector search only
    python eval.py --rerank --tag v3       # add the cross-encoder reranker, saved to eval/results/v3.json
    python eval.py --rerank --expand       # also add LLM query expansion (nondeterministic: run it twice)
    python eval.py --show-misses           # also print the top 3 chunks of every miss
    python eval.py --compare base after    # side-by-side of two saved runs
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import config
from chunking import chunk_text
from parsing import parse_document


# ── Scoring (pure functions, unit-tested) ───────────────────────────────

def normalize(text: str) -> str:
    """Lowercase and keep only a-z0-9, so '$ 95.00' matches '$95.00' and curly quotes never matter."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def is_hit(item: dict, chunk_text_: str, doc_id: str | None) -> bool:
    if item.get("expected_doc") and doc_id != item["expected_doc"]:
        return False
    return normalize(item["must_contain"]) in normalize(chunk_text_)


def first_hit_rank(item: dict, results: list[dict]) -> int | None:
    """1-based rank of the first relevant chunk, or None."""
    for rank, r in enumerate(results, start=1):
        if is_hit(item, r["text"], r["metadata"].get("doc_id")):
            return rank
    return None


def summarize(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "recall": 0.0, "mrr": 0.0}
    return {
        "n": n,
        "recall": sum(r is not None and r <= k for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r is not None and r <= k) / n,
    }


# ── Runs ────────────────────────────────────────────────────────────────

def load_golden() -> list[dict]:
    return json.loads(Path(config.GOLDEN_PATH).read_text(encoding="utf-8"))


def run_eval(retriever, k: int, expand: bool, rerank: bool = False) -> dict:
    """`retriever` is any object with retrieve(...) and expand_query(...) (the real Retriever, or a fake)."""
    golden = load_golden()

    if expand and not retriever.expand_query(golden[0]["question"]):
        raise SystemExit(
            "Query expansion returned nothing: the LLM call is failing "
            "(wrong model name? missing key?). Fix it, or run without --expand."
        )

    rows, by_cat = [], defaultdict(list)
    for item in golden:
        # Retrieve deeper than k so a miss tells us "rank 8" instead of just "not in top 5".
        # Metrics still use k only (see summarize).
        results = retriever.retrieve(item["question"], top_k=config.MAX_TOP_K,
                           deal_id=item.get("deal_id"), expand=expand, rerank=rerank)
        rank = first_hit_rank(item, results)
        by_cat[item.get("category", "other")].append(rank)
        rows.append({
            "id": item["id"],
            "category": item.get("category"),
            "rank": rank,                 # rank within the retrieved list, None if absent
            "retrieved": len(results),    # how deep we actually looked
            "top": [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["metadata"]["doc_id"],
                    "score": round(r["score"], 3),
                    "rerank_score": round(r["rerank_score"], 3) if "rerank_score" in r else None,
                    "preview": r["text"][:160],
                }
                for r in results[:3]
            ],
        })

    all_ranks = [r["rank"] for r in rows]
    return {
        "config": {
            "k": k, "expand": expand, "rerank": rerank,
            "llm_model": config.LLM_MODEL,
            "rerank_model": config.RERANK_MODEL_NAME if rerank else None,
            "chunk_size": config.CHUNK_SIZE, "overlap": config.CHUNK_OVERLAP,
            "model": config.EMBEDDING_MODEL_NAME,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "overall": summarize(all_ranks, k),
        "by_category": {c: summarize(r, k) for c, r in by_cat.items()},
        "questions": rows,
    }


def print_report(report: dict, show_misses: bool = False) -> None:
    k = report["config"]["k"]
    print(f"\nconfig: {report['config']}")
    o = report["overall"]
    print(f"\nOVERALL  recall@{k} = {o['recall']:.2f}   MRR = {o['mrr']:.2f}   (n={o['n']})")
    for cat, s in sorted(report["by_category"].items()):
        print(f"  {cat:<8} recall@{k} = {s['recall']:.2f}   MRR = {s['mrr']:.2f}   (n={s['n']})")

    misses = [q for q in report["questions"] if q["rank"] is None or q["rank"] > k]
    print(f"\nMISSES ({len(misses)}):")
    for q in misses:
        where = (f"found at rank {q['rank']}" if q["rank"]
                 else f"NOT in the {q.get('retrieved', '?')} chunks retrieved")
        print(f"  - {q['id']:<22} ({q['category']})  {where}")
        if show_misses:
            for t in q.get("top", []):
                chunk_no = t["chunk_id"].split("__")[-1]
                rr = f" rr={t['rerank_score']:.2f}" if t.get("rerank_score") is not None else ""
                print(f"      {t['score']:.3f}{rr}  {t['doc_id']}/{chunk_no}: {t['preview']}")


def save_report(report: dict, tag: str) -> Path:
    out_dir = Path(config.EVAL_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def compare(tag_a: str, tag_b: str) -> None:
    a, b = (json.loads((Path(config.EVAL_RESULTS_DIR) / f"{t}.json").read_text()) for t in (tag_a, tag_b))
    print(f"\n{'':<10}{tag_a:>12}{tag_b:>12}{'delta':>10}")
    for name, sa, sb in [("overall", a["overall"], b["overall"])] + [
        (c, a["by_category"][c], b["by_category"].get(c, {"recall": 0, "mrr": 0}))
        for c in a["by_category"]
    ]:
        for metric in ("recall", "mrr"):
            print(f"{name + ' ' + metric:<22}{sa[metric]:>10.2f}{sb[metric]:>12.2f}{sb[metric] - sa[metric]:>+10.2f}")


def check_golden() -> bool:
    """Sanity check without any model: is each expected answer inside ONE chunk?

    If not, retrieval can never succeed on that question — the chunker cut the answer.
    """
    chunks_by_doc = {}
    for doc in config.PRELOAD_DOCUMENTS:
        chunks_by_doc[doc["doc_id"]] = chunk_text(parse_document(doc["file_path"]))
        print(f"{doc['doc_id']}: {len(chunks_by_doc[doc['doc_id']])} chunks")

    ok = True
    for item in load_golden():
        docs = [item["expected_doc"]] if item.get("expected_doc") else list(chunks_by_doc)
        found = sum(normalize(item["must_contain"]) in normalize(c)
                    for d in docs for c in chunks_by_doc[d])
        status = "OK     " if found else "MISSING"
        ok &= bool(found)
        print(f"  {status} {item['id']:<24} in {found} chunk(s)")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--expand", action="store_true", help="enable LLM query expansion (one extra LLM call per question)")
    p.add_argument("--rerank", action="store_true", help="rerank candidates with the cross-encoder")
    p.add_argument("--show-misses", action="store_true", help="print the top 3 chunks of each miss")
    p.add_argument("--tag", help="save the report as eval/results/<tag>.json")
    p.add_argument("--check-golden", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = p.parse_args()

    if args.check_golden:
        raise SystemExit(0 if check_golden() else 1)
    if args.compare:
        compare(*args.compare)
        return

    from container import get_container  # imported here so `--check-golden` needs no model / API key

    report = run_eval(get_container().retriever, args.k, expand=args.expand, rerank=args.rerank)
    print_report(report, show_misses=args.show_misses)
    if args.tag:
        print(f"\nsaved -> {save_report(report, args.tag)}")


if __name__ == "__main__":
    main()
