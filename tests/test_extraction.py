import pytest

from extraction import (FIELD_QUERIES, ExtractionError, Extractor, build_field_context,
                        clean_fields, parse_field_result)
from tests.fakes import FakeLLM, FakeRetriever, make_chunk, write_html


def extractor(retriever=None, llm=None, **options):
    return Extractor(retriever or FakeRetriever(), llm or FakeLLM(), **options)


# ── queries ─────────────────────────────────────────────────────────────

def test_known_fields_never_call_the_llm():
    llm = FakeLLM()
    queries = extractor(llm=llm).resolve_queries(["deal_value", "buyer"])
    assert queries["deal_value"] == FIELD_QUERIES["deal_value"]
    assert "Parent" not in queries["buyer"]  # defined terms attract every clause that mentions them
    assert llm.json_calls == []


def test_unknown_fields_get_one_llm_written_query_each_and_it_is_cached():
    llm = FakeLLM(json_fn=lambda s, u: {"queries": {"outside_date": "written outside", "break_fee": "written fee"}})
    ex = extractor(llm=llm)
    first = ex.resolve_queries(["deal_value", "outside_date", "break_fee"])
    assert first["outside_date"] == "outside date. written outside"  # own words + expansion
    assert len(llm.json_calls) == 1                                    # ONE call for both unknown fields
    assert "Fields: outside_date, break_fee" in llm.json_calls[0][1]
    ex.resolve_queries(["outside_date"])
    assert len(llm.json_calls) == 1                                    # cached


def test_unknown_field_falls_back_to_its_own_words_when_the_llm_fails():
    assert extractor(llm=FakeLLM()).resolve_queries(["outside_date"]) == {"outside_date": "outside date"}


def test_non_string_written_queries_are_ignored():
    llm = FakeLLM(json_fn=lambda s, u: {"queries": {"x_y": 42}})
    assert extractor(llm=llm).resolve_queries(["x_y"]) == {"x_y": "x y"}


def test_query_cache_is_per_extractor_not_global():
    llm = FakeLLM(json_fn=lambda s, u: {"queries": {"x_y": "written"}})
    extractor(llm=llm).resolve_queries(["x_y"])
    assert extractor(llm=FakeLLM()).resolve_queries(["x_y"]) == {"x_y": "x y"}


# ── fields ──────────────────────────────────────────────────────────────

def test_clean_fields_dedupes_and_strips():
    assert clean_fields([" a ", "b", "a", ""]) == ["a", "b"]


def test_clean_fields_rejects_empty_and_too_many():
    with pytest.raises(ValueError):
        clean_fields(["  ", ""])
    with pytest.raises(ValueError):
        clean_fields([f"f{i}" for i in range(100)])
    with pytest.raises(ValueError):
        clean_fields(["a", "b", "c"], max_fields=2)


# ── prompt + parsing ────────────────────────────────────────────────────

def test_build_field_context_labels_chunks_per_field():
    context, chunk_of = build_field_context("price", "per share price",
                                            [make_chunk("c1", "alpha"), make_chunk("c2", "beta")])
    assert set(chunk_of) == {"S1", "S2"}
    assert "FIELD: price" in context and "WHAT TO LOOK FOR: per share price" in context
    assert "[S1] alpha" in context and "[S2] beta" in context


def test_build_field_context_without_evidence():
    context, chunk_of = build_field_context("price", "q", [])
    assert chunk_of == {} and "(no excerpt found)" in context


def test_parse_maps_labels_to_sources_and_drops_fake_ones():
    chunk_of = {"S1": make_chunk("c1", doc_id="merger")}
    out = parse_field_result({"value": "$95.00", "evidence": ["S1", "S99"]}, chunk_of)
    assert out == {"value": "$95.00", "sources": [{"chunk_id": "c1", "doc_id": "merger"}]}


def test_parse_accepts_bare_values_lists_and_string_evidence():
    assert parse_field_result("Microsoft", {}) == {"value": "Microsoft", "sources": []}
    out = parse_field_result({"value": ["A", "B"], "evidence": "S1"}, {"S1": make_chunk("c1")})
    assert out["value"] == ["A", "B"] and len(out["sources"]) == 1


@pytest.mark.parametrize("empty", [None, "", [], "None", "null", "N/A", " not stated ", "Unknown"])
def test_parse_treats_empty_and_null_like_values_as_null(empty):
    out = parse_field_result({"value": empty, "evidence": ["S1"]}, {"S1": make_chunk("c1")})
    assert out == {"value": None, "sources": []}


def test_parse_keeps_real_values_that_start_like_null_words():
    assert parse_field_result({"value": "National Amusements"}, {})["value"] == "National Amusements"


# ── orchestration ───────────────────────────────────────────────────────

def test_one_retrieval_and_one_llm_call_per_field():
    retriever = FakeRetriever(chunks_fn=lambda q: [make_chunk(f"c_{q[:4]}", f"text for {q}", "merger")])
    llm = FakeLLM(json_fn=lambda s, u: {"queries": {"break_fee": "written"}} if "search query" in s
                  else {"value": "v", "evidence": ["S1"]})
    ex = Extractor(retriever, llm, chunks_per_field=4, pool=100, max_workers=2)
    run = ex.run("d", ["deal_value", "break_fee"])

    assert [c["query"] for c in retriever.calls] == [FIELD_QUERIES["deal_value"], "break fee. written"]
    assert all(c["top_k"] == 4 and c["deal_id"] == "d" and c["expand"] is False and c["pool"] == 100
               for c in retriever.calls)
    field_calls = [u for s, u in llm.json_calls if "ONE field" in s]
    assert len(field_calls) == 2
    assert sum("FIELD: break_fee" in u and "FIELD: deal_value" in u for u in field_calls) == 0  # fields never mixed
    assert run.results["break_fee"]["sources"] == [{"chunk_id": "c_brea", "doc_id": "merger"}]


def test_field_without_evidence_skips_the_llm():
    llm = FakeLLM(json_fn=lambda s, u: pytest.fail("LLM called"))
    run = extractor(retriever=FakeRetriever(), llm=llm).run("d", ["deal_value"])
    assert run.results["deal_value"] == {"value": None, "sources": []}


def test_llm_failure_names_the_failing_fields():
    retriever = FakeRetriever(chunks_fn=lambda q: [make_chunk()])
    llm = FakeLLM(json_fn=lambda s, u: None if "FIELD: buyer" in u else {"value": "x", "evidence": []})
    with pytest.raises(ExtractionError, match="buyer"):
        extractor(retriever, llm).extract("d", ["deal_value", "buyer"])


def test_per_field_override_of_chunk_count():
    retriever = FakeRetriever()
    extractor(retriever, chunks_per_field=4).retrieve_evidence("d", ["deal_value"], per_field=20)
    assert retriever.calls[0]["top_k"] == 20


def test_end_to_end_with_real_chroma(container, tmp_path):
    f = write_html(tmp_path / "d.html", "The merger consideration is $95.00 in cash per share.")
    container.ingestor.ingest(f, deal_id="d", doc_id="merger")
    container.llm.json_fn = lambda s, u: {"value": "$95.00", "evidence": ["S1"]} if "$95.00" in u else None
    out = container.extractor.extract("d", ["deal_value"])
    assert out["deal_value"] == {"value": "$95.00",
                                 "sources": [{"chunk_id": "d__merger__0", "doc_id": "merger"}]}


def test_run_exposes_intermediates():
    retriever = FakeRetriever(chunks_fn=lambda q: [make_chunk("c1", "the price is $95.00")])
    llm = FakeLLM(json_fn=lambda s, u: {"value": "$95.00", "evidence": ["S1"]})
    run = extractor(retriever, llm).run("d", ["deal_value"])
    assert run.queries["deal_value"] == FIELD_QUERIES["deal_value"]
    assert run.evidence["deal_value"][0]["chunk_id"] == "c1"
    assert "[S1] the price is $95.00" in run.contexts["deal_value"]
    assert run.raw["deal_value"] == {"value": "$95.00", "evidence": ["S1"]}
