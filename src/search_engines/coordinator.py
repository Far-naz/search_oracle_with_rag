import os
from typing import List, Optional

from config import ENABLE_LLM_SEARCH
from src.advisors.models import MatchAdvisor
from src.helpers.text_normalization import strip_filler, strip_query_wrapper
from src.intent.hybrid_recognizer import HybridIntentRecognizer
from src.intent.types import Intent, IntentPrediction
from src.search_engines.chroma_engine import ChromaSearchEngine
from src.search_engines.llm_search import llm_search_advisors

DEFAULT_LLM_ESCALATION_MIN_SCORE = 0.65


class AdvisorSearchCoordinator:

    def __init__(
        self,
        search_engine: ChromaSearchEngine,
        intent_recognizer: HybridIntentRecognizer,
        enable_llm_search: bool = ENABLE_LLM_SEARCH,
        llm_escalation_min_score: float = DEFAULT_LLM_ESCALATION_MIN_SCORE,
    ):
        self.engine = search_engine
        self.intent = intent_recognizer
        self.enable_llm_search = enable_llm_search
        self.llm_escalation_min_score = llm_escalation_min_score
        self.last_prediction: IntentPrediction | None = None
        self.last_llm_attempted = False
        self.last_llm_error: str | None = None

    def search(
        self,
        raw_query: str,
        top_k: int = 5,
        api_key: str | None = None,
    ) -> List[MatchAdvisor]:
        self.last_llm_attempted = False
        self.last_llm_error = None

        prediction = self.intent.predict(raw_query)
        self.last_prediction = prediction
        intent = Intent.parse(prediction.intent)

        slots = prediction.slots
        section_filter = slots.get("section")
        topic = slots.get("topic")

        if topic:
            search_query = topic
        else:
            search_query = strip_query_wrapper(
                raw_query
            )  # ← catches slot extraction failures

        if intent == Intent.LIST_ALL:
            advisors = self.engine.get_all_advisors()
            return [
                MatchAdvisor(advisor=a, score=1.0, document="")
                for a in advisors.values()
            ][:top_k]

        if prediction.intent == Intent.ADVISOR_SEARCH:
            if slots.get("name"):
                return self.engine._search_hard_scores(slots["name"])
            return self.engine.search(
                search_query, top_k=top_k, section_filter=section_filter
            )

        if prediction.intent == Intent.TOPIC_SEARCH:
            return self.engine.search(
                search_query, top_k=top_k, section_filter=section_filter
            )

        if intent == Intent.SEARCH_BY_NAME:
            # Hard name lookup only - no semantic noise
            results = self.engine._search_hard_scores(raw_query)
            if results:
                return results[:top_k]
            # Graceful fallback if name wasn't found
            return self.engine.search(raw_query, top_k=top_k)

        if intent == Intent.SEARCH_BY_SECTION:
            section = self._extract_section(raw_query)
            return self.engine.search(raw_query, top_k=top_k, section_filter=section)

        if intent in {
            Intent.PUBLICATION_OR_EXPERTISE_SEARCH,
            Intent.SEARCH_BY_EXPERTISE
        }:
            # Strip conversational filler before embedding
            cleaned = strip_filler(raw_query)
            return self.engine.search(cleaned, top_k=top_k)

        # "get_info", "unknown", fallback
        results = self.engine.search(raw_query, top_k=top_k)
        if intent == Intent.UNKNOWN:
            return self._escalate_unknown_intent_if_weak(
                raw_query=raw_query,
                standard_results=results,
                top_k=top_k,
                api_key=api_key,
            )

        return results

    def _escalate_unknown_intent_if_weak(
        self,
        raw_query: str,
        standard_results: List[MatchAdvisor],
        top_k: int,
        api_key: str | None,
    ) -> List[MatchAdvisor]:
        if not self.enable_llm_search or not self._results_are_weak(standard_results):
            return standard_results

        llm_api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not llm_api_key:
            self.last_llm_error = "Missing API key or advisor data."
            return standard_results

        self.last_llm_attempted = True
        llm_results, llm_error = llm_search_advisors(
            query=raw_query,
            advisors=list(self.engine.get_all_advisors().values()),
            top_k=top_k,
            api_key=llm_api_key,
        )
        self.last_llm_error = llm_error

        if llm_results:
            return llm_results

        return standard_results

    def _results_are_weak(self, results: List[MatchAdvisor]) -> bool:
        if not results:
            return True

        best_score = max(result.score for result in results)
        return best_score < self.llm_escalation_min_score

    def _extract_section(self, query: str) -> Optional[str]:
        """Pull a section name out of the query if present."""
        known_sections = ["systems", "theory", "ai", "security", "networks"]
        q = query.lower()
        for section in known_sections:
            if section in q:
                return section
        return None
