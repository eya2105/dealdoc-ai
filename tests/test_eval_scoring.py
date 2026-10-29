from eval import normalize, is_hit, first_hit_rank, summarize


def res(text, doc="merger_agreement"):
    return {"text": text, "metadata": {"doc_id": doc}}


def test_normalize_ignores_spacing_and_punctuation():
    assert normalize("$ 95.00") == normalize("$95.00")
    assert normalize("“ Superior Proposal ” means") == normalize("Superior Proposal means")


def test_expected_doc_is_enforced():
    item = {"must_contain": "45.3%", "expected_doc": "proxy_statement"}
    assert not is_hit(item, "premium of 45.3%", "merger_agreement")
    assert is_hit(item, "premium of 45.3%", "proxy_statement")


def test_first_hit_rank():
    item = {"must_contain": "April 28, 2022"}
    results = [res("nothing"), res("meeting on April 28, 2022"), res("April 28, 2022")]
    assert first_hit_rank(item, results) == 2
    assert first_hit_rank(item, [res("nothing")]) is None


def test_summarize():
    s = summarize([1, 2, None, 6], k=5)
    assert s["n"] == 4
    assert s["recall"] == 0.5          # ranks 1 and 2 are within k=5
    assert s["mrr"] == (1 + 0.5) / 4   # rank 6 is outside k, so it counts as a miss
