"""
Advisor repository
==================
Loads advisor profiles from disk. Keeps data access separate from indexing and UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from advisors.models import Advisor
from config import ADVISOR_DATA_FILE


DEFAULT_ADVISORS_PATH = Path(ADVISOR_DATA_FILE)


def load_available_advisors(data_path: str | Path = DEFAULT_ADVISORS_PATH) -> List[Advisor]:
    """Load advisor records from the JSON source file."""
    data_file = Path(data_path)
    with data_file.open("r", encoding="utf-8") as file:
        records = json.load(file)

    advisors: List[Advisor] = []
    for item in records:
        advisors.append(
            Advisor(
                name=item["name"],
                title=item["title"],
                section=item["section"],
                email=item.get("email", ""),
                profile_url=item.get("profile_url", ""),
                research_output=item.get("research_output", []),
                activities=item.get("activities", []),
                press_media=item.get("press_media", []),
            )
        )

    return advisors
