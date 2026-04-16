"""
Enrich advisor JSON records from their CBS profile URLs.

This script:
- loads advisors from the JSON file
- fetches each advisor profile page
- extracts missing fields from the HTML and embedded structured data
- merges only missing values into the JSON records
- writes the updated JSON back to disk

Usage:
    python enrich_advisors_from_profiles.py
    python enrich_advisors_from_profiles.py --input data/cbs_employees.json --output data/cbs_employees.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from html import unescape
from pathlib import Path
from typing import Any, Iterable

import requests

from advisor_repository import load_available_advisors
from advisors_data import Advisor
from config import ADVISOR_DATA_FILE


DEFAULT_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = _normalize_whitespace(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def _first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        cleaned = _normalize_whitespace(value)
        if cleaned:
            return cleaned
    return ""


def _is_count_like(value: str) -> bool:
    cleaned = _normalize_whitespace(value)
    if not cleaned:
        return True
    if re.fullmatch(r"\(?\d+\)?", cleaned):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return True
    if re.fullmatch(r"\(\d+\)\s*\d+", cleaned):
        return True
    return False


def _is_placeholder_heading(value: str) -> bool:
    normalized = _normalize_whitespace(value).lower()
    return normalized in {
        "research output",
        "activities",
        "press/media",
        "press media",
        "press & media",
    }


def _is_meaningful_title(value: str) -> bool:
    cleaned = _normalize_whitespace(value)
    if not cleaned or _is_count_like(cleaned):
        return False
    if _is_placeholder_heading(cleaned):
        return False

    # Ignore very short fragments that are usually UI noise.
    if len(cleaned) < 8:
        return False

    # Require at least one alphabetic character.
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", cleaned):
        return False

    return True


def _clean_section_titles(values: Iterable[str]) -> list[str]:
    filtered = [value for value in values if _is_meaningful_title(value)]
    return _dedupe_preserve_order(filtered)


def _normalize_section_values(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    return _clean_section_titles(list(values))


def _extract_titles_from_links(html_text: str, content_path: str) -> list[str]:
    """Extract title text from anchors pointing to content pages.

    Example content_path values:
    - "/en/publications/"
    - "/en/activities/"
    - "/en/clippings/"
    """
    anchor_pattern = re.compile(
        rf"<a[^>]+href=(['\"])(?P<href>[^'\"]*{re.escape(content_path)}[^'\"]*)\1[^>]*>(?P<title>.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    titles: list[str] = []

    for match in anchor_pattern.finditer(html_text):
        href = unescape(match.group("href"))
        if "?" in href:
            # These are typically filters or counters, not title pages.
            continue

        title_raw = re.sub(r"<[^>]+>", " ", match.group("title"))
        title = _normalize_whitespace(unescape(title_raw))

        if title.lower().startswith("view all"):
            continue
        if title.lower().startswith("share on"):
            continue
        if title.lower() in {"pure", "plumx metrics detail page", "open access", "file"}:
            continue

        titles.append(title)

    return _clean_section_titles(titles)


def _list_is_count_only(values: list[str]) -> bool:
    if not values:
        return False
    return all(_is_count_like(value) for value in values)


def _extract_meta_tags(html_text: str) -> dict[str, list[str]]:
    meta_pattern = re.compile(
        r'<meta\s+[^>]*(?:name|property|itemprop)=(["\'])(?P<key>[^"\']+)\1[^>]*content=(["\'])(?P<value>.*?)\3[^>]*>',
        re.IGNORECASE | re.DOTALL,
    )
    tags: dict[str, list[str]] = {}
    for match in meta_pattern.finditer(html_text):
        key = unescape(match.group("key")).strip().lower()
        value = _normalize_whitespace(unescape(match.group("value")))
        tags.setdefault(key, []).append(value)
    return tags


def _extract_json_ld_blocks(html_text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    script_pattern = re.compile(
        r'<script[^>]*type=(["\'])application/ld\+json\1[^>]*>(?P<body>.*?)</script>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in script_pattern.finditer(html_text):
        raw_body = match.group("body").strip()
        if not raw_body:
            continue
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            blocks.append(parsed)
        elif isinstance(parsed, list):
            blocks.extend(item for item in parsed if isinstance(item, dict))
    return blocks


def _extract_visible_text_snippets(html_text: str) -> list[str]:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html_text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>|</div>|</li>|</tr>|</h[1-6]>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    lines = [_normalize_whitespace(line) for line in text.splitlines()]
    return [line for line in lines if line]


def _collect_section_values(lines: list[str], labels: list[str]) -> list[str]:
    collected: list[str] = []
    label_patterns = [re.compile(rf"^{re.escape(label)}\s*:?\s*(.*)$", re.IGNORECASE) for label in labels]

    for index, line in enumerate(lines):
        for pattern in label_patterns:
            match = pattern.match(line)
            if not match:
                continue

            inline_value = _normalize_whitespace(match.group(1))
            if inline_value:
                collected.append(inline_value)
                break

            next_index = index + 1
            while next_index < len(lines):
                candidate = lines[next_index]
                if candidate:
                    if len(candidate) > 3:
                        collected.append(candidate)
                    break
                next_index += 1
            break

    return _dedupe_preserve_order(collected)


def _extract_section_items(lines: list[str], headings: list[str]) -> list[str]:
    collected: list[str] = []
    heading_patterns = [re.compile(rf"^{re.escape(heading)}\s*:?\s*$", re.IGNORECASE) for heading in headings]
    bullet_pattern = re.compile(r"^[•\-*\u2022]\s+(.*)$")
    separator_pattern = re.compile(r"^(?:[A-Z][A-Za-z0-9\s/&(),.'-]{2,}|\d+\.)$")

    for index, line in enumerate(lines):
        if not any(pattern.match(line) for pattern in heading_patterns):
            continue

        next_index = index + 1
        while next_index < len(lines):
            candidate = lines[next_index]
            if not candidate:
                next_index += 1
                continue

            if any(pattern.match(candidate) for pattern in heading_patterns):
                break

            bullet_match = bullet_pattern.match(candidate)
            if bullet_match:
                collected.append(bullet_match.group(1))
                next_index += 1
                continue

            if separator_pattern.match(candidate) and candidate.lower() not in {h.lower() for h in headings}:
                break

            collected.append(candidate)
            next_index += 1

    return _clean_section_titles(collected)


def _extract_from_json_ld(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}

    for block in blocks:
        block_type = block.get("@type")
        if isinstance(block_type, list):
            block_type = " ".join(str(item) for item in block_type)
        block_type = str(block_type or "").lower()

        if "person" not in block_type and "faculty" not in block_type and "professor" not in block_type:
            continue

        name = block.get("name")
        if name and not extracted.get("name"):
            extracted["name"] = _normalize_whitespace(str(name))

        job_title = block.get("jobTitle") or block.get("honorificPrefix")
        if job_title and not extracted.get("title"):
            extracted["title"] = _normalize_whitespace(str(job_title))

        email = block.get("email")
        if email and not extracted.get("email"):
            extracted["email"] = _normalize_whitespace(str(email)).replace("mailto:", "")

        address = block.get("address")
        if isinstance(address, dict) and not extracted.get("building"):
            address_text = _first_nonempty(
                str(address.get(key, "")) for key in ("streetAddress", "addressLocality", "addressRegion")
            )
            if address_text:
                extracted["building"] = address_text

        same_as = block.get("sameAs")
        if same_as and not extracted.get("profile_url"):
            if isinstance(same_as, list):
                extracted["profile_url"] = _first_nonempty(str(item) for item in same_as)
            else:
                extracted["profile_url"] = _normalize_whitespace(str(same_as))

        keywords = block.get("keywords")
        if keywords and not extracted.get("research_interests"):
            if isinstance(keywords, list):
                extracted["research_interests"] = _dedupe_preserve_order(str(item) for item in keywords)
            else:
                extracted["research_interests"] = _dedupe_preserve_order(
                    part for part in re.split(r"[,;|]", str(keywords))
                )

    return extracted


def scrape_profile(profile_url: str) -> dict[str, Any]:
    response = requests.get(
        profile_url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()

    html_text = response.text
    meta_tags = _extract_meta_tags(html_text)
    json_ld_blocks = _extract_json_ld_blocks(html_text)
    visible_lines = _extract_visible_text_snippets(html_text)
    publication_titles = _extract_titles_from_links(html_text, "/en/publications/")
    activity_titles = _extract_titles_from_links(html_text, "/en/activities/")
    press_media_titles = _extract_titles_from_links(html_text, "/en/clippings/")

    extracted: dict[str, Any] = {}
    extracted.update(_extract_from_json_ld(json_ld_blocks))

    if not extracted.get("name"):
        extracted["name"] = _first_nonempty(
            meta_tags.get("citation_author", [])
            + meta_tags.get("og:title", [])
            + [
                _normalize_whitespace(re.sub(r"\s*\|.*$", "", title))
                for title in meta_tags.get("title", [])
            ]
        )

    if not extracted.get("title"):
        extracted["title"] = _first_nonempty(
            meta_tags.get("citation_title", [])
            + meta_tags.get("og:title", [])
        )

    if not extracted.get("email"):
        email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", html_text, re.IGNORECASE)
        if email_match:
            extracted["email"] = email_match.group(0)

    if not extracted.get("building"):
        building_candidates = _collect_section_values(
            visible_lines,
            ["Address", "Office", "Room", "Building", "Location"],
        )
        if building_candidates:
            extracted["building"] = building_candidates[0]

    if not extracted.get("research_interests"):
        research_candidates = _collect_section_values(
            visible_lines,
            ["Research interests", "Research interest", "Research areas", "Interests", "Areas of interest"],
        )
        if research_candidates:
            extracted["research_interests"] = research_candidates

    if not extracted.get("courses"):
        course_candidates = _collect_section_values(
            visible_lines,
            ["Teaching", "Courses", "Courses taught", "Course"],
        )
        if course_candidates:
            extracted["courses"] = course_candidates

    if not extracted.get("recent_publications"):
        publication_candidates = _collect_section_values(
            visible_lines,
            ["Selected publications", "Publications", "Recent publications", "Publications and papers"],
        )
        if publication_candidates:
            extracted["recent_publications"] = publication_candidates[:10]

    if not extracted.get("supervised_topics"):
        topic_candidates = _collect_section_values(
            visible_lines,
            ["Supervision", "Supervised topics", "Thesis topics", "Student supervision"],
        )
        if topic_candidates:
            extracted["supervised_topics"] = topic_candidates[:10]

    if not extracted.get("research_output"):
        research_output_candidates = publication_titles or _extract_section_items(
            visible_lines,
            ["Research output"],
        )
        if research_output_candidates:
            extracted["research_output"] = research_output_candidates[:20]

    if not extracted.get("activities"):
        activities_candidates = activity_titles or _extract_section_items(
            visible_lines,
            ["Activities"],
        )
        if activities_candidates:
            extracted["activities"] = activities_candidates[:20]

    if not extracted.get("press_media"):
        press_media_candidates = press_media_titles or _extract_section_items(
            visible_lines,
            ["Press/Media", "Press Media", "Press & Media"],
        )
        if press_media_candidates:
            extracted["press_media"] = press_media_candidates[:20]

    return extracted


def merge_missing_fields(advisor: Advisor, scraped: dict[str, Any]) -> Advisor:
    """Return a copy of advisor with only missing fields filled from scraped data."""
    current_research_output = _normalize_section_values(advisor.research_output)
    current_activities = _normalize_section_values(advisor.activities)
    current_press_media = _normalize_section_values(advisor.press_media)

    scraped_research_output = _normalize_section_values(scraped.get("research_output", []))
    scraped_activities = _normalize_section_values(scraped.get("activities", []))
    scraped_press_media = _normalize_section_values(scraped.get("press_media", []))

    return Advisor(
        name=advisor.name or scraped.get("name", ""),
        title=advisor.title or scraped.get("title", ""),
        section=advisor.section,
        email=advisor.email or scraped.get("email", ""),
        profile_url=advisor.profile_url or scraped.get("profile_url", ""),
        research_interests=advisor.research_interests or scraped.get("research_interests", []),
        courses=advisor.courses or scraped.get("courses", []),
        recent_publications=advisor.recent_publications or scraped.get("recent_publications", []),
        supervised_topics=advisor.supervised_topics or scraped.get("supervised_topics", []),
        building=advisor.building or scraped.get("building", ""),
        research_output=current_research_output or scraped_research_output,
        activities=current_activities or scraped_activities,
        press_media=current_press_media or scraped_press_media,
    )


def update_json_file(input_path: str | Path, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the JSON file, enrich each advisor, and save the updated data."""
    input_file = Path(input_path)
    output_file = Path(output_path) if output_path else input_file

    advisors = load_available_advisors(input_file)
    with input_file.open("r", encoding="utf-8") as file:
        records = json.load(file)

    updated_records: list[dict[str, Any]] = []
    for record, advisor in zip(records, advisors):
        profile_url = record.get("profile_url", "").strip()
        if not profile_url:
            updated_records.append(record)
            continue

        try:
            scraped = scrape_profile(profile_url)
        except Exception as exc:
            print(f"Skipping {advisor.name}: {exc}")
            updated_records.append(record)
            continue

        enriched = merge_missing_fields(advisor, scraped)
        enriched_record = dict(record)
        enriched_record.update(asdict(enriched))
        updated_records.append(enriched_record)
        print(f"Updated {enriched.name}")

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(updated_records, file, ensure_ascii=False, indent=2)

    return updated_records


def fetch_and_update_advisors() -> None:
    parser = argparse.ArgumentParser(description="Enrich advisor JSON data from CBS profile pages.")
    parser.add_argument("--input", default=ADVISOR_DATA_FILE, help="Path to the input JSON file.")
    parser.add_argument(
        "--output",
        default=ADVISOR_DATA_FILE,
        help="Path to write the enriched JSON file. Defaults to the input file.",
    )
    args = parser.parse_args()

    update_json_file(args.input, args.output)
    #print(f"Finished writing enriched advisor data to {args.output}")


#if __name__ == "__main__":
#    fetch_and_update_advisors()
