from dataclasses import dataclass, field
from typing import Dict, List
import json

from helper import safe_json_loads


@dataclass
class Advisor:
    name: str
    title: str
    section: str
    email: str
    profile_url: str
    research_output: List[str] = field(default_factory=list)
    activities: List[str] = field(default_factory=list)
    press_media: List[str] = field(default_factory=list)


def build_advisor_document(advisor: Advisor) -> str:
    """
    Flatten an advisor profile into a single searchable text document.
    This is the unit we index for retrieval.
    """
    parts = [
        f"{advisor.name} — {advisor.title}, {advisor.section}",
        f"Email: {advisor.email}",
        f"Profile URL: {advisor.profile_url}",
        f"Research output: {', '.join(advisor.research_output)}",
    ]
    if advisor.research_output:
        parts.append(f"Research output: {'; '.join(advisor.research_output)}")
    if advisor.activities:
        parts.append(f"Activities: {'; '.join(advisor.activities)}")
    if advisor.press_media:
        parts.append(f"Press/Media: {'; '.join(advisor.press_media)}")
    return "\n".join(parts)


def build_advisor_metadata(advisor: Advisor) -> Dict[str, str]:
    return {
        "name": advisor.name,
        "title": advisor.title,
        "section": advisor.section,
        "email": advisor.email,
        "profile_url": advisor.profile_url,
        "research_output": json.dumps(advisor.research_output),
        "activities": json.dumps(advisor.activities),
        "press_media": json.dumps(advisor.press_media),
    }


def reconstruct_advisor(metadata: Dict[str, str]) -> Advisor:
    return Advisor(
        name=metadata.get("name", ""),
        title=metadata.get("title", ""),
        section=metadata.get("section", ""),
        email=metadata.get("email", ""),
        profile_url=metadata.get("profile_url", ""),
        research_output=safe_json_loads(metadata.get("research_output")),
        activities=safe_json_loads(metadata.get("activities")),
        press_media=safe_json_loads(metadata.get("press_media")),
    )
