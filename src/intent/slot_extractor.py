import re
from typing import Dict

SLOT_PATTERNS = {
    #"topic": [
    #    re.compile(r"\bin\s+([a-z][a-z\s]{2,30}?)(?:\s+(?:field|area|research|department|section))?$", re.I),
    #    re.compile(r"\b(?:about|on|studying|working on|interested in)\s+([a-z][a-z\s]{2,30})", re.I),
    #    re.compile(r"\bfor\s+(?:my\s+)?(?:thesis|research)\s+(?:on|in|about)\s+([a-z][a-z\s]{2,30})", re.I),
    #],
    "topic": [
        re.compile(r"\bin\s+([a-z][a-z\s]{2,40})", re.I),
        re.compile(r"\b(?:about|on|studying|working on|interested in)\s+([a-z][a-z\s]{2,40})", re.I),
        re.compile(r"\bfor\s+(?:my\s+)?(?:thesis|research)\s+(?:on|in|about)\s+([a-z][a-z\s]{2,40})", re.I),
    ],
    "section": [
        re.compile(r"\b(?:in|from|within)\s+the\s+([a-z]+)\s+(?:section|department|group|division)", re.I),
        re.compile(r"\b([a-z]+)\s+(?:section|department|group|division)\b", re.I),
    ],
    "name": [
        # Capitalized words after "find/contact/about" - likely a name
        re.compile(r"\b(?:find|contact|about|email)\s+(?:prof\.?|dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"),
        re.compile(r"\b(?:prof\.?|dr\.?|professor)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"),
    ],
}

def extract_slots(text: str) -> Dict[str, str]:
    slots = {}
    for slot_name, patterns in SLOT_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                slots[slot_name] = match.group(1).strip().lower()
                break  # first match wins per slot type
    return slots

TRAILING_NOISE = {"please", "now", "today", "area", "field", "research", "section", "department"}

def clean_slot(value: str) -> str:
    tokens = value.strip().split()
    while tokens and tokens[-1].lower() in TRAILING_NOISE:
        tokens.pop()
    return " ".join(tokens)