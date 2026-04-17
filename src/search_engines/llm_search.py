import json
import re
from typing import List, Tuple

import requests

from src.advisors.match_output import MatchAdvisor
from src.advisors.models import Advisor, build_advisor_document


def _extract_openrouter_text(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "\n".join(text_parts)

    return ""


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

    target = name.strip().lower()
    if not target:
        return None

    for advisor in advisors:
        if advisor.name.strip().lower() == target:
            return advisor

    for advisor in advisors:
        candidate = advisor.name.strip().lower()
        if target in candidate or candidate in target:
            return advisor

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
    for advisor in advisors:
        advisor_lines.append(
            " | ".join(
                [
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
    {{"name": "Exact Advisor Name", "score": 0.0, "reason": "one short reason"}}
  ]
}}
Rules:
- Choose at most {top_k} advisors.
- Use advisor names exactly as listed.
- Score must be between 0 and 1.
- Do not include markdown, prose, or code fences.

Student query:
{query}

Advisor candidates:
{chr(10).join(advisor_lines)}
"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={
                "model": "openrouter/auto",
                "messages": [
                    {"role": "system", "content": "You rank advisor matches and return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            },
            timeout=30,
        )
        if response.status_code != 200:
            return [], f"LLM request failed with status {response.status_code}."

        data = response.json()
        text = _extract_openrouter_text(data)
        if not text:
            return [], "LLM response did not contain usable text."
    except Exception:
        return [], "LLM request failed due to network or response parsing error."

    parsed = _parse_matches_json(text)
    if not parsed:
        return [], "LLM response was not valid match JSON."

    selected: List[MatchAdvisor] = []
    seen_names = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue

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
