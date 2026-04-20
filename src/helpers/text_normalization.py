from __future__ import annotations

import re


def normalize_name(name: str) -> str:
    if not name:
        return ""

    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower())
    return re.sub(r"\s+", " ", normalized).strip()