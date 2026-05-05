from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

IntentSource = Literal["rule", "ml", "fallback"]


class Intent(str, Enum):
    ADVISOR_SEARCH = "advisor_search"
    AVAILABILITY_SEARCH = "availability_search"
    TOPIC_SEARCH = "topic_search"
    PUBLICATION_OR_EXPERTISE_SEARCH = "publication_or_expertise_search"
    SEARCH_BY_EXPERTISE = "search_by_expertise"
    SEARCH_BY_NAME = "search_by_name"
    LIST_ALL = "list_all"
    SEARCH_BY_SECTION = "search_by_section"
    GET_INFO = "get_info"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | Intent) -> Intent | None:
        if isinstance(value, cls):
            return value

        try:
            return cls(value)
        except ValueError:
            return None


SUPPORTED_INTENTS = {
    intent.value
    for intent in (
        Intent.ADVISOR_SEARCH,
        Intent.AVAILABILITY_SEARCH,
        Intent.TOPIC_SEARCH,
        Intent.PUBLICATION_OR_EXPERTISE_SEARCH,
        Intent.SEARCH_BY_EXPERTISE,
        Intent.SEARCH_BY_NAME,
        Intent.LIST_ALL,
        Intent.SEARCH_BY_SECTION,
    )
}


@dataclass
class IntentPrediction:
    intent: Intent | str
    confidence: float
    source: IntentSource
    slots: dict[str, str] = field(default_factory=dict)
