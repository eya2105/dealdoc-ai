import pytest

import eval as ev
from tests.fakes import FakeRetriever, make_chunk


def retriever_that_finds_every_answer():
    golden = {g["question"]: g for g in ev.load_golden()}

    def chunks(question):
        item = golden[question]
        return [make_chunk("x__y__1", "unrelated"),
                make_chunk("x__y__2", item["must_contain"], doc_id=item.get("expected_doc", "any"))]

    return FakeRetriever(chunks_fn=chunks)


def test_run_eval_scores_a_retriever_that_ranks_the_answer_second():
    report = ev.run_eval(retriever_that_finds_every_answer(), k=5, expand=False)
    assert report["overall"]["recall"] == 1.0
    assert report["overall"]["mrr"] == pytest.approx(0.5)
    assert report["questions"][0]["top"][0]["chunk_id"] == "x__y__1"
    assert report["config"]["rerank"] is False


def test_run_eval_passes_the_flags_to_the_retriever():
    retriever = retriever_that_finds_every_answer()
    ev.run_eval(retriever, k=5, expand=False, rerank=True)
    assert all(c["rerank"] is True and c["expand"] is False for c in retriever.calls)


def test_run_eval_refuses_to_run_with_broken_expansion():
    with pytest.raises(SystemExit):
        ev.run_eval(FakeRetriever(variants=[]), k=5, expand=True)
