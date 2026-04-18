import re
from collections import Counter
from typing import List

from config import STOPWORDS


_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    use_lemmatization: bool = True,
) -> list[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if use_lemmatization:
        tokens = [_normalize_token(t) for t in tokens]

    if remove_stopwords:
        filtered = [t for t in tokens if t not in STOPWORDS]
        if filtered:
            return filtered
    return tokens


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_doc_len: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    tf = Counter(doc_tokens)
    score = 0.0
    for token in query_tokens:
        f = tf.get(token, 0)
        if f == 0:
            continue
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
        score += numerator / denominator
    max_score = len(set(query_tokens)) * (k1 + 1)
    return min(score / max_score, 1.0) if max_score > 0 else 0.0
