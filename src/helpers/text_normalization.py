from __future__ import annotations

import re


def normalize_name(name: str) -> str:
    if not name:
        return ""

    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def strip_filler(query: str) -> str:
    """Remove conversational wrapper before semantic search."""
    filler = [
        "i'm looking for", "i need", "find me", "can you find",
        "advisor who is working on", "advisor who works on",
        "advisor working on", "who is working on", "who works on",
        "someone who does", "an advisor for", "advisor for",
        "i am looking for"
    ]
    q = query.lower().strip()
    changed = True
    while changed:
        changed = False
        for phrase in filler:
            if not q.startswith(phrase):
                continue
            q = q[len(phrase):].strip()
            changed = True
            break
    return q


# Ordered — most specific patterns first
QUERY_WRAPPER_PATTERNS = [
    re.compile(r"^(?:find|looking for|i need|i want|recommend|suggest)\s+(?:a\s+|an\s+)?(?:supervisor|advisor|professor|teacher)\s+(?:in|for|about|on|working on|who (?:works on|researches|specializes in))\s+", re.I),
    re.compile(r"^(?:find|looking for|i need|i want)\s+(?:a\s+|an\s+)?(?:supervisor|advisor|professor)\s+", re.I),
    re.compile(r"^(?:find|search for|show me|get me)\s+", re.I),
    re.compile(r"^(?:who (?:works on|researches|is working on|specializes in))\s+", re.I),
]

def strip_query_wrapper(query: str) -> str:
    """Remove conversational/structural wrapper, leaving only the core topic."""
    q = query.strip()
    for pattern in QUERY_WRAPPER_PATTERNS:
        cleaned = pattern.sub("", q).strip()
        if cleaned and cleaned != q:
            return cleaned
    return q
