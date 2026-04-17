from typing import Tuple

import requests


def extract_openrouter_text(data: dict) -> str:
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


def openrouter_chat_completion(
    prompt: str,
    api_key: str,
    system_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1200,
    model: str = "openrouter/auto",
) -> Tuple[str | None, str | None]:
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )

        if response.status_code != 200:
            return None, f"LLM request failed with status {response.status_code}."

        text = extract_openrouter_text(response.json())
        if not text:
            return None, "LLM response did not contain usable text."

        return text, None
    except Exception:
        return None, "LLM request failed due to network or response parsing error."
