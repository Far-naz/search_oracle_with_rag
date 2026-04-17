import os
from typing import Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


from advisor_match_output import MatchAdvisor
from advisors_data import (
    Advisor,
    build_advisor_document,
    build_advisor_metadata,
    reconstruct_advisor,
)
from config import CLEAR_DB, LOWEST_SEMANTIC_SCORE
from bm25 import tokenize, bm25_score


class ChromaSearchEngine:

    def __init__(
        self,
        collection_name: str = "advisors",
        persist_directory: str = "./chroma_data",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        os.makedirs(persist_directory, exist_ok=True)

        if CLEAR_DB:
            tmp = chromadb.PersistentClient(path=persist_directory)
            try:
                tmp.delete_collection(name=collection_name)
            except Exception:
                pass

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.advisors: Dict[str, Advisor] = {}
        self.advisor_documents: Dict[str, str] = {}
        self._load_cached_advisors()

    # ── Cache ──────────────────────────────────────────────────────────────

    def _load_cached_advisors(self) -> None:
        if self.collection.count() == 0:
            return
        stored = self.collection.get(include=["documents", "metadatas"])
        for index, advisor_id in enumerate(stored.get("ids", [])):
            metadata = stored["metadatas"][index] or {}
            document = (
                stored["documents"][index] if index < len(stored["documents"]) else ""
            )
            self.advisors[advisor_id] = reconstruct_advisor(metadata)
            self.advisor_documents[advisor_id] = document

    # ── Ingestion ──────────────────────────────────────────────────────────

    def add_advisors(self, advisors: List[Advisor], replace: bool = True) -> None:
        if replace and self.collection.count() > 0:
            self.client.delete_collection(name=self.collection_name)
            # FIX: always pass embedding_function when recreating
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self.advisors.clear()
            self.advisor_documents.clear()

        for index, advisor in enumerate(advisors):
            advisor_id = f"advisor_{index}_{advisor.name.replace(' ', '_')}"
            document_text = build_advisor_document(advisor)
            metadata = build_advisor_metadata(advisor)
            self.collection.add(
                ids=[advisor_id],
                documents=[document_text],
                metadatas=[metadata],
            )
            self.advisors[advisor_id] = advisor
            self.advisor_documents[advisor_id] = document_text

    def _build_candidates(self, results) -> List[MatchAdvisor]:
        """Convert raw ChromaDB results into MatchAdvisor objects."""
        matched: List[MatchAdvisor] = []
        if not (results and results.get("ids") and results["ids"]):
            return matched

        for index, advisor_id in enumerate(results["ids"][0]):
            # cosine distance → similarity (ChromaDB returns 0..2, clamp to 0..1)
            distance = results["distances"][0][index]
            semantic_score = max(0.0, min(1.0, 1.0 - distance))

            advisor = self.advisors.get(advisor_id)
            if not advisor:
                meta = (results.get("metadatas") or [[]])[0][index]
                if meta:
                    advisor = reconstruct_advisor(meta)
                    self.advisors[advisor_id] = advisor

            if not advisor:
                continue

            doc = self.advisor_documents.get(advisor_id)
            if not doc and results.get("documents"):
                doc = results["documents"][0][index]
                self.advisor_documents[advisor_id] = doc

            if not doc:
                continue
            matched.append(
                MatchAdvisor(advisor=advisor, score=semantic_score, document=doc)
            )

        return matched

    def _bm25_rerank(
        self,
        query: str,
        candidates: List[MatchAdvisor],
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> List[MatchAdvisor]:
        query_tokens = tokenize(query)
        all_docs = [tokenize(item.document or "") for item in candidates]
        avg_doc_len = sum(len(d) for d in all_docs) / max(len(all_docs), 1)

        for item, doc_tokens in zip(candidates, all_docs):
            bm25 = bm25_score(query_tokens, doc_tokens, avg_doc_len)
            item.score = semantic_weight * item.score + bm25_weight * bm25

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _search_hard_scores(
        self, query: str, section_filter: Optional[str] = None
    ) -> List[MatchAdvisor]:
        exact_ids = []
        first_name = None
        email = None
        words = query.split()
        for word in words:
            if "@" in word:
                email = word
            elif word.isalpha():
                first_name = word

        if email:
            for advisor_id, advisor in self.advisors.items():
                if advisor.email.lower() == email.lower():
                    exact_ids.append(advisor_id)

        if first_name:
            for advisor_id, advisor in self.advisors.items():
                if advisor.name.lower().find(first_name.lower()) != -1:
                    exact_ids.append(advisor_id)

        exact_matches: List["MatchAdvisor"] = []
        for advisor_id in exact_ids:
            advisor = self.advisors.get(advisor_id)
            if not advisor:
                continue
            if section_filter and advisor.section != section_filter:
                continue
            exact_matches.append(
                MatchAdvisor(
                    advisor=advisor,
                    score=1.0,
                    document=self.advisor_documents.get(advisor_id, ""),
                )
            )
        return exact_matches
    # ── Public API ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[str] = None,
    ) -> List[MatchAdvisor]:

        exact_matches = self._search_hard_scores(query, section_filter)
        if len(exact_matches) >= 1:
            return exact_matches[:1]

        if self.collection.count() == 0:
            print("Warning: Collection is empty.")
            return []

        candidate_k = min(self.collection.count(), max(top_k * 10, 30))

        results = self.collection.query(
            query_texts=[query],
            n_results=candidate_k,
            where={"section": {"$eq": section_filter}} if section_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        candidates = self._build_candidates(results)
        candidates = self._bm25_rerank(query, candidates)

        selected_candidates = [
            c for c in candidates if c.score >= LOWEST_SEMANTIC_SCORE
        ]
        return selected_candidates[:top_k]

    def get_all_advisors(self) -> Dict[str, Advisor]:
        return self.advisors

    def get_collection_stats(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "total_advisors": self.collection.count(),
            "persist_directory": self.persist_directory,
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.advisors.clear()
        self.advisor_documents.clear()
