import sys
from types import SimpleNamespace

import pytest

import config
import eval_extract
from eval_extract import (classify, evidence_has_answer, first_answer_rank, score_field, stage_of)
from extraction import ExtractionError, ExtractionRun
from tests.fakes import make_chunk


# ── pure scoring ────────────────────────────────────────────────────────

def test_correct_and_grounded():
    assert score_field(["95.00"], "$95.00 in cash", ["merger consideration $ 95.00"]) == (True, True)


def test_correct_but_not_grounded():
    assert score_field(["95.00"], "$95.00", ["unrelated chunk"]) == (True, False)


def test_wrong_value():
    assert score_field(["95.00"], "$90.00", ["$95.00"]) == (False, False)


def test_null_value():
    assert score_field(["95.00"], None, []) == (False, False)


def test_list_values_and_multiple_expected():
    assert score_field(["Allen", "Goldman"], ["Allen & Company", "Goldman Sachs"], ["Allen", "Goldman Sachs"]) == (True, True)


def test_evidence_has_answer():
    assert evidence_has_answer(["95.00"], ["nothing", "price is $ 95.00"])
    assert not evidence_has_answer(["95.00"], ["nothing"])
    assert not evidence_has_answer(["a", "b"], ["only a here"])


def test_classify_blames_the_right_stage():
    assert classify(True, True) == "ok"
    assert classify(False, True).startswith("LLM miss")
    assert classify(False, False).startswith("retrieval miss")


def test_stage_of():
    assert stage_of(3, 3, True) == "ok"
    assert stage_of(1, 3, True) == "flaky"
    assert stage_of(0, 3, True) == "LLM"
    assert stage_of(0, 3, False) == "retrieval"


def test_first_answer_rank():
    chunks = [{"text": "nothing"}, {"text": "price $ 95.00 here"}, {"text": "95.00"}]
    assert first_answer_rank(["95.00"], chunks) == 2
    assert first_answer_rank(["absent"], chunks) is None
    assert first_answer_rank(["a", "b"], [{"text": "a"}, {"text": "a and b"}]) == 2  # ALL strings in ONE chunk


# ── the script, with a fake extractor ───────────────────────────────────

def fake_container(extractor, store=None):
    return SimpleNamespace(extractor=extractor, store=store or SimpleNamespace(get_texts=lambda ids: ["price $ 95.00"] * len(ids)))


class FakeExtractor:
    def __init__(self, answers=None, error=None):
        self.answers, self.error = answers or {}, error
        self.evidence_calls = []

    def run(self, deal_id, fields):
        if self.error:
            raise self.error
        results = {f: {"value": self.answers.get(f), "sources": [{"chunk_id": "d__m__0", "doc_id": "m"}] if self.answers.get(f) else []}
                   for f in fields}
        raw = {f: {"value": results[f]["value"], "evidence": []} for f in fields}
        evidence = {f: [make_chunk("d__m__0", "price $ 95.00")] for f in fields}
        return ExtractionRun(fields, {f: f"q {f}" for f in fields}, evidence, {f: "ctx" for f in fields}, raw, results)

    def retrieve_evidence(self, deal_id, fields, per_field=3):
        self.evidence_calls.append(per_field)
        chunk = make_chunk("d__m__3", "price $ 95.00")
        return fields, {f: f"q {f}" for f in fields}, {f: [chunk] for f in fields}


def test_main_prints_one_line_per_field_and_the_totals(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eval_extract.py"])
    monkeypatch.setattr(eval_extract, "_container", lambda: fake_container(FakeExtractor({"deal_value": "$95.00"})))
    eval_extract.main()
    out = capsys.readouterr().out
    assert "deal_value" in out and "1/1" in out
    assert "correct 1/8" in out


def test_main_reports_extraction_errors_without_a_traceback(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval_extract.py"])
    monkeypatch.setattr(eval_extract, "_container",
                        lambda: fake_container(FakeExtractor(error=ExtractionError("no JSON for: buyer"))))
    with pytest.raises(SystemExit) as exc:
        eval_extract.main()
    assert "no JSON for: buyer" in str(exc.value) and "rate limit" in str(exc.value)


def test_retrieval_only_never_runs_the_extraction(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eval_extract.py", "--retrieval-only"])
    extractor = FakeExtractor()
    monkeypatch.setattr(eval_extract, "_container", lambda: fake_container(extractor))
    eval_extract.main()
    out = capsys.readouterr().out
    assert "deal_value: answer chunk first at rank 1" in out
    # the fake returns the same chunk for every field, so only the field expecting "95.00" matches
    assert f"answer within the {config.EXTRACT_CHUNKS_PER_FIELD} chunks the LLM sees: 1/8 fields" in out
    assert extractor.evidence_calls == [config.EXTRACT_CHUNKS_PER_FIELD]


def test_depth_is_passed_to_the_retrieval(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval_extract.py", "--retrieval-only", "--depth", "20"])
    extractor = FakeExtractor()
    monkeypatch.setattr(eval_extract, "_container", lambda: fake_container(extractor))
    eval_extract.main()
    assert extractor.evidence_calls == [20]
