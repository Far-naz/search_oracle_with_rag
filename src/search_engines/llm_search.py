import json
import re
from difflib import SequenceMatcher
from typing import List, Tuple

from src.advisors.models import Advisor, MatchAdvisor, build_advisor_document
from src.helpers.openrouter_client import openrouter_chat_completion
from src.helpers.text_normalization import normalize_name


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_matches_json(text: str) -> list[dict]:
    cleaned = _strip_code_fences(text)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(payload, dict):
        items = payload.get("matches", [])
        return items if isinstance(items, list) else []

    if isinstance(payload, list):
        return payload

    return []


def _to_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


def _resolve_advisor(name: str, advisors: List[Advisor]) -> Advisor | None:
    if not name:
        return None

    target = normalize_name(name)
    if not target:
        return None

    normalized = {normalize_name(advisor.name): advisor for advisor in advisors}
    if target in normalized:
        return normalized[target]

    best_match: Advisor | None = None
    best_score = 0.0
    second_best_score = 0.0

    for advisor in advisors:
        candidate = normalize_name(advisor.name)
        ratio = SequenceMatcher(None, target, candidate).ratio()
        if ratio > best_score:
            second_best_score = best_score
            best_score = ratio
            best_match = advisor
        elif ratio > second_best_score:
            second_best_score = ratio

    # Only accept a fuzzy name if it is clearly close and unambiguous.
    if best_match and best_score >= 0.9 and (best_score - second_best_score) >= 0.03:
        return best_match

    return None


def _parse_advisor_id(value, advisor_count: int) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None

    if 1 <= parsed <= advisor_count:
        return parsed
    return None


def llm_search_advisors(
    query: str,
    advisors: List[Advisor],
    top_k: int = 5,
    api_key: str | None = None,
) -> Tuple[List[MatchAdvisor], str | None]:
    if not api_key or not advisors:
        return [], "Missing API key or advisor data."

    advisor_lines = []
    for index, advisor in enumerate(advisors, start=1):
        advisor_lines.append(
            " | ".join(
                [
                    f"id={index}",
                    f"name={advisor.name}",
                    f"title={advisor.title}",
                    f"section={advisor.section}",
                    f"email={advisor.email}",
                    f"research_output={'; '.join(advisor.research_output[:4])}",
                    f"activities={'; '.join(advisor.activities[:3])}",
                    f"press_media={'; '.join(advisor.press_media[:3])}",
                ]
            )
        )

    prompt = f"""You are ranking advisor matches for a student query.
Return ONLY valid JSON in this format:
{{
  "matches": [
        {{"id": 0, "name": "Exact Advisor Name", "score": 0.0, "reason": "one short reason"}}
  ]
}}
Rules:
- Choose at most {top_k} advisors.
- Use advisor id exactly as listed in candidates.
- Use advisor names exactly as listed.
- Score must be between 0 and 1.
- Do not include markdown, prose, or code fences.

Student query:
{query}

Advisor candidates:
{chr(10).join(advisor_lines)}
"""

    text, error = openrouter_chat_completion(
        prompt=prompt,
        api_key=api_key,
        system_prompt="You rank advisor matches and return strict JSON only.",
        temperature=0.1,
        max_tokens=1200,
    )
    if error:
        return [], error

    if not text:
        return [], "LLM did not return any text."
    parsed = _parse_matches_json(text)
    if not parsed:
        return [], "LLM response was not valid match JSON."

    selected: List[MatchAdvisor] = []
    seen_names = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue

        advisor = None
        advisor_id = _parse_advisor_id(item.get("id"), advisor_count=len(advisors))
        if advisor_id is not None:
            advisor = advisors[advisor_id - 1]

        if advisor is None:
            advisor = _resolve_advisor(str(item.get("name", "")), advisors)

        if not advisor or advisor.name in seen_names:
            continue

        seen_names.add(advisor.name)
        reason = str(item.get("reason", "")).strip()
        selected.append(
            MatchAdvisor(
                advisor=advisor,
                score=_to_score(item.get("score")),
                document=reason or build_advisor_document(advisor),
            )
        )

        if len(selected) >= top_k:
            break

    if not selected:
        return [], "LLM returned matches that did not map to advisor names."

    return selected, None
