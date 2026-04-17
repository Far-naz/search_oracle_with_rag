from src.search_engines.bm25 import bm25_score, tokenize


def test_tokenize_removes_stopwords_by_default() -> None:
    tokens = tokenize("This is a test about energy systems")
    assert "is" not in tokens
    assert "a" not in tokens
    assert "energy" in tokens
    assert "systems" in tokens


def test_tokenize_falls_back_to_original_when_all_stopwords() -> None:
    tokens = tokenize("the and to")
    assert tokens == ["the", "and", "to"]


def test_tokenize_can_keep_stopwords() -> None:
    tokens = tokenize("the energy system", remove_stopwords=False)
    assert tokens == ["the", "energy", "system"]


def test_bm25_score_empty_inputs() -> None:
    assert bm25_score([], ["energy"], 3.0) == 0.0
    assert bm25_score(["energy"], [], 3.0) == 0.0


def test_bm25_score_returns_normalized_value() -> None:
    query = ["energy", "system"]
    doc = ["energy", "energy", "system", "analysis"]
    score = bm25_score(query, doc, avg_doc_len=4.0)
    assert 0.0 < score <= 1.0
