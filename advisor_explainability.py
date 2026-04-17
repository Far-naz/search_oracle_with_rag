"""
Advisor explainability helpers
==============================
Contains the optional Gemini explanation prompt plus highlighting utilities.
"""

from __future__ import annotations

import html
import re
from typing import List

import requests

from advisors_data import Advisor
from bm25 import tokenize


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _tokenize_text(value: str) -> set[str]:
    if not value:
        return set()

    token = tokenize(value, remove_stopwords=True)
    return set(token)


def get_matched_terms(query: str, advisor: Advisor) -> List[str]:
    matched_terms: List[str] = []
    query_terms = _tokenize_text(query)

    if query_terms & _tokenize_text(advisor.name):
        matched_terms.append(f"Name: {advisor.name}")
    if query_terms & _tokenize_text(advisor.title):
        matched_terms.append(f"Title: {advisor.title}")
    if query_terms & _tokenize_text(advisor.section):
        matched_terms.append(f"Section: {advisor.section}")
    if advisor.email and query.lower() in advisor.email.lower():
        matched_terms.append(f"Email: {advisor.email}")

    for output in advisor.research_output:
        if _tokenize_text(output) & query_terms:
            matched_terms.append(f"Research Output: {output}")
    for activity in advisor.activities:
        if _tokenize_text(activity) & query_terms:
            matched_terms.append(f"Activity: {activity}")
    for media in advisor.press_media:
        if _tokenize_text(media) & query_terms:
            matched_terms.append(f"Press/Media: {media}")

    return matched_terms


def get_highlight_terms(query: str, advisor: Advisor) -> List[str]:
    query_terms = list(_tokenize_text(query))
    highlight_terms: List[str] = []

    def collect_matches(text: str) -> None:
        field_terms = _tokenize_text(text)
        for term in query_terms:
            if term in field_terms:
                highlight_terms.append(term)

    collect_matches(advisor.name)
    collect_matches(advisor.title)
    collect_matches(advisor.section)

    if advisor.email and query.lower() in advisor.email.lower():
        highlight_terms.extend(_tokenize_text(query))

    for output in advisor.research_output:
        collect_matches(output)
    for activity in advisor.activities:
        collect_matches(activity)
    for media in advisor.press_media:
        collect_matches(media)

    return _ordered_unique(highlight_terms)


def highlight_html(text: str, terms: List[str]) -> str:
    if not text or not terms:
        return html.escape(text)

    highlighted = html.escape(text)
    for term in sorted(_ordered_unique(terms), key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"(?i)\b({re.escape(html.escape(term))})\b")
        highlighted = pattern.sub(r'<mark class="highlight-term">\1</mark>', highlighted)
    return highlighted


def render_term_pills(terms: List[str]) -> str:
    pills = []
    for term in _ordered_unique(terms):
        pills.append(f'<span class="match-pill">{html.escape(term)}</span>')
    return "".join(pills)


def generate_rag_explanation(query: str, results: list[dict], api_key: str | None = None):
    if not api_key:
        return None

    profile_text = ""
    for i, result in enumerate(results):
        advisor: Advisor = result["advisor"]
        matched_terms = result.get("matched_terms") or get_matched_terms(query, advisor)
        profile_text += f"Advisor {i + 1}:\n"
        profile_text += f"Name: {advisor.name}\n"
        profile_text += f"section: {advisor.section}\n"
        profile_text += f'research_output: {", ".join(advisor.research_output)}\n'
        profile_text += f'activities: {", ".join(advisor.activities)}\n'
        profile_text += f'press_media: {", ".join(advisor.press_media)}\n'
        profile_text += f"Matched Terms: {', '.join(matched_terms)}\n\n"

    prompt = f"""Given the following student query and matched advisor profiles, explain in natural language why each advisor is a good
match for the student's research interests and goals.
Student Query: {query}
Here are the top matching advisor profiles:
{profile_text}

For each advisor, write a concise 2-3 sentence explanation of why they
are a good match for this student's interests. Mention specific
publications, courses, or research areas that align. If the advisor has
limited availability, note this. Be encouraging but honest.

Format your response as:
### 1. [Advisor Name]
[Explanation]

### 2. [Advisor Name]
[Explanation]
"""
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 1500,
                    "temperature": 0.7,
                },
            },
            timeout=30,
        )
        if response.status_code != 200:
            return None
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None
