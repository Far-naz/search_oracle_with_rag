import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern

from src.intent.types import IntentPrediction


@dataclass
class IntentRule:
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[Pattern[str]] = field(default_factory=list)


def _compile_pattern(pattern: str) -> Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE)


DEFAULT_RULES: Dict[str, IntentRule] = {
    "advisor_search": IntentRule(
        keywords=[
            "advisor",
            "supervisor",
            "professor",
            "teacher",
            "recommend",
            "match",
            "thesis",
            "research topic",
            "find professor",
        ],
        regex_patterns=[
            _compile_pattern(r"\b(find|recommend|match)\b.*\b(advisor|supervisor|professor)\b"),
            _compile_pattern(r"\b(advisor|supervisor|professor)\b.*\b(for|about|in)\b"),
            _compile_pattern(r"\b(find|looking for|need|want)\b.{0,20}\b(supervisor|advisor|professor)\b.{0,10}\bin\b"),
        ],
    ),
    "topic_search": IntentRule(
        keywords=[
            "topic",
            "field",
            "area",
            "domain",
            "research area",
            "interested in",
            "exploring",
            "learn about",
            "teaching",
            "lecturing",
            "working on",
        ],
        regex_patterns=[
            _compile_pattern(r"\b(explore|interested in|learn about|working on)\b.*\b(topic|field|area|research)\b"),
            _compile_pattern(r"\bwhat\b.*\btopics?\b.*\bavailable\b"),
        ],
    ),
    "availability_search": IntentRule(
        keywords=[
            "available",
            "availability",
            "accepting students",
            "open slots",
            "taking students",
            "supervision capacity",
            "currently accepting",
        ],
        regex_patterns=[
            _compile_pattern(r"\b(accepting|taking)\b.*\b(students|advisees)\b"),
            _compile_pattern(r"\b(available|availability|open)\b.*\b(now|currently|this semester)\b"),
        ],
    ),
    "publication_or_expertise_search": IntentRule(
        keywords=[
            "publication",
            "published",
            "paper",
            "expertise",
            "method",
            "technique",
            "working on",
            "specialist",
        ],
        regex_patterns=[
            _compile_pattern(r"\b(publication|published|paper)\b.*\b(on|about|in)\b"),
            _compile_pattern(r"\b(expertise|expert|specialist|technique|method)\b"),
            _compile_pattern(r"\b(advisor|supervisor|professor)\b.*\b(working on|works on|researching|specializes in)\b"),
        ],
    ),
}


class RuleBasedIntentRecognizer:
    def __init__(self, rules: Dict[str, IntentRule] | None = None) -> None:
        self.rules = rules or DEFAULT_RULES

    def predict(self, text: str) -> IntentPrediction | None:
        normalized = (text or "").strip().lower()
        if not normalized:
            return None

        best_intent: str | None = None
        best_score = 0.0

        for intent, rule in self.rules.items():
            keyword_hits = sum(1 for keyword in rule.keywords if keyword in normalized)
            regex_hits = sum(1 for pattern in rule.regex_patterns if pattern.search(normalized))

            if keyword_hits == 0 and regex_hits == 0:
                continue

            # Regex hits are stronger signals than keyword matches.
            score = min(1.0, (keyword_hits * 0.2) + (regex_hits * 0.5))
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent is None:
            return None

        return IntentPrediction(intent=best_intent, confidence=best_score, source="rule")
