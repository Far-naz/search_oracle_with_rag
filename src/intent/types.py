from dataclasses import dataclass, field
from typing import Literal

IntentSource = Literal["rule", "ml", "fallback"]
SUPPORTED_INTENTS = {
    "advisor_search",
    "availability_search",
    "topic_search",
    "publication_or_expertise_search",
    "search_by_name",
    "list_all",
    "search_by_section",
}

@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    source: str
    slots: dict[str, str] = field(default_factory=dict)
