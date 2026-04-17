"""
Chroma index bootstrap
======================
Connects the advisor repository to the Chroma search engine.
"""

from __future__ import annotations

from advisors.repository import load_available_advisors
from search_engines.chroma_engine import ChromaSearchEngine


def initialize_chroma_database(
    collection_name: str = "advisors",
    persist_directory: str = "./chroma_data",
    refresh: bool = True,
) -> ChromaSearchEngine:
    """Build or refresh the persisted Chroma collection from advisor records."""
    advisors = load_available_advisors()
    engine = ChromaSearchEngine(collection_name=collection_name, persist_directory=persist_directory)

    if refresh or engine.collection.count() == 0:
        engine.add_advisors(advisors, replace=True)

    return engine
