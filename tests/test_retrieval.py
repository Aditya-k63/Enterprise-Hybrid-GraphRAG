import pytest
from unittest.mock import patch, MagicMock
from app.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_empty():
    result = reciprocal_rank_fusion([])
    assert result == []


def test_rrf_single_list():
    items = [
        {"content": "doc1", "score": 0.9},
        {"content": "doc2", "score": 0.8},
    ]
    result = reciprocal_rank_fusion([items])
    assert len(result) == 2
    assert result[0]["rrf_score"] >= result[1]["rrf_score"]


def test_rrf_merges_duplicates():
    list1 = [{"content": "shared doc", "score": 0.9}]
    list2 = [{"content": "shared doc", "score": 0.8}]
    result = reciprocal_rank_fusion([list1, list2])
    assert len(result) == 1
    assert result[0]["rrf_score"] > 1 / 61


def test_rrf_fusion_ordering():
    list1 = [{"content": "a", "score": 0.9}, {"content": "b", "score": 0.5}]
    list2 = [{"content": "b", "score": 0.9}, {"content": "c", "score": 0.5}]
    result = reciprocal_rank_fusion([list1, list2])
    contents = [r["content"] for r in result]
    assert "b" == contents[0]
