import pytest

from ids import slugify, make_chunk_id
from store import build_where


def test_slugify_normalizes():
    assert slugify("MSFT / Activision Merger!") == "msft_activision_merger"


def test_slugify_neutralizes_html():
    assert "<" not in slugify("<img src=x onerror=alert(1)>")


def test_slugify_rejects_empty():
    with pytest.raises(ValueError):
        slugify("  !!! ")


def test_chunk_ids_unique_across_docs_and_deals():
    assert make_chunk_id("d1", "doc", 0) != make_chunk_id("d2", "doc", 0)
    assert make_chunk_id("d", "a", 0) != make_chunk_id("d", "b", 0)


def test_build_where_variants():
    assert build_where() is None
    assert build_where(deal_id="d") == {"deal_id": "d"}
    assert build_where(doc_id="x") == {"doc_id": "x"}
    assert build_where("d", "x") == {"$and": [{"deal_id": "d"}, {"doc_id": "x"}]}
