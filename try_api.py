"""Smoke-test a RUNNING DealDoc AI server by calling its real URLs.

    python try_api.py                       # server on http://127.0.0.1:8000
    python try_api.py --base http://localhost:8000
    python try_api.py --skip-extract        # /extract takes about a minute on the free Groq tier

Start the server first (uvicorn main:app --reload) and index the documents (python ingest.py).
"""
import argparse
import sys
import time

import httpx

results: list[bool] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))


def call(client: httpx.Client, method: str, path: str, **kwargs) -> tuple[httpx.Response, float]:
    started = time.time()
    response = client.request(method, path, **kwargs)
    elapsed = time.time() - started
    print(f"\n{method} {client.base_url}{path}  ->  {response.status_code}  ({elapsed:.1f}s)")
    return response, elapsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="http://127.0.0.1:8000", help="server address")
    p.add_argument("--deal", default="msft_activision", help="deal_id to query")
    p.add_argument("--question", default="What is the purchase price per share?")
    p.add_argument("--skip-extract", action="store_true", help="skip the slow /extract call")
    args = p.parse_args()

    with httpx.Client(base_url=args.base, timeout=300) as client:
        try:
            r, _ = call(client, "GET", "/health")
        except httpx.ConnectError:
            sys.exit(f"Cannot reach {args.base}. Start the server first: uvicorn main:app --reload")
        check("health is ok", r.status_code == 200 and r.json().get("status") == "ok", r.text)

        r, _ = call(client, "GET", "/documents")
        docs = r.json().get("documents", []) if r.status_code == 200 else []
        check("documents are indexed", r.status_code == 200 and len(docs) > 0,
              ", ".join(f"{d['deal_id']}/{d['doc_id']}: {d['chunks']} chunks" for d in docs) or "none: run python ingest.py")

        r, _ = call(client, "POST", "/query", json={"question": args.question, "top_k": 5, "deal_id": args.deal})
        body = r.json() if r.status_code == 200 else {}
        check("query returns an answer", r.status_code == 200 and bool(body.get("answer")))
        check("query returns numbered sources",
              [s.get("rank") for s in body.get("sources", [])] == list(range(1, len(body.get("sources", [])) + 1))
              and len(body.get("sources", [])) > 0)
        print(f"  answer : {body.get('answer', r.text)[:300]}")
        for s in body.get("sources", [])[:3]:
            print(f"  source {s['rank']}: {s['doc_id']} score={s['score']}  {s['text_preview'][:80]}")

        if not args.skip_extract:
            fields = ["deal_value", "buyer", "target"]
            r, elapsed = call(client, "POST", "/extract", json={"deal_id": args.deal, "fields": fields})
            got = r.json().get("extracted_fields", {}) if r.status_code == 200 else {}
            check("extract returns every requested field", r.status_code == 200 and set(got) == set(fields), r.text[:200])
            for name, res in got.items():
                print(f"  {name:<12} = {res['value']!r}   sources: {[s['chunk_id'] for s in res['sources']]}")

        print("\nError handling:")
        r, _ = call(client, "POST", "/extract", json={"deal_id": "no_such_deal"})
        check("unknown deal gives 404", r.status_code == 404)
        r, _ = call(client, "POST", "/query", json={"question": "x", "top_k": 999})
        check("top_k out of range gives 422", r.status_code == 422)
        r, _ = call(client, "POST", "/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
        check("unsupported file type gives 400", r.status_code == 400)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()