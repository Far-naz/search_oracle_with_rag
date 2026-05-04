from __future__ import annotations

from typing import List, Tuple


IntentExample = Tuple[str, str]


DEFAULT_INTENT_TRAINING_EXAMPLES: List[IntentExample] = [
    ("recommend an advisor for machine learning", "advisor_search"),
    ("find a supervisor for my thesis", "advisor_search"),
    ("match me with a professor for finance", "advisor_search"),
    ("I need an advisor for digital platforms", "advisor_search"),
    ("I am exploring research fields in economics", "topic_search"),
    ("what topic in causal inference should I study", "topic_search"),
    ("I want to learn about entrepreneurship research", "topic_search"),
    ("which research area covers labor markets", "topic_search"),
    ("which advisors are accepting students now", "availability_search"),
    ("who is currently available to supervise", "availability_search"),
    ("find professors taking thesis students", "availability_search"),
    ("show advisors with open supervision capacity", "availability_search"),
    ("who has publications on diffusion models", "publication_or_expertise_search"),
    ("looking for expertise in time-series forecasting", "publication_or_expertise_search"),
    ("advisor who is working on labor", "publication_or_expertise_search"),
    ("professor researching sustainable finance", "publication_or_expertise_search"),
    ("find Alice Smith", "search_by_name"),
    ("show me the profile for Bob Jones", "search_by_name"),
    ("who is Chen Li", "search_by_name"),
    ("search for professor Maria Hansen", "search_by_name"),
    ("list all advisors", "list_all"),
    ("show every advisor in the database", "list_all"),
    ("give me the full advisor list", "list_all"),
    ("display all supervisors", "list_all"),
    ("advisors in the AI section", "search_by_section"),
    ("find professors from security", "search_by_section"),
    ("show supervisors in networks", "search_by_section"),
    ("who belongs to the systems group", "search_by_section"),
]


def split_training_examples(
    examples: List[IntentExample] | None = None,
) -> tuple[list[str], list[str]]:
    selected_examples = examples or DEFAULT_INTENT_TRAINING_EXAMPLES
    texts = [text for text, _intent in selected_examples]
    intents = [intent for _text, intent in selected_examples]
    return texts, intents
