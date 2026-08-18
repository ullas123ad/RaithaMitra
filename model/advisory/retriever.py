"""
RaithaMitra Local Agricultural Knowledge Retriever.

Provides lightweight, CPU-efficient, deterministic lexical knowledge retrieval (RAG)
grounded on verified Indian agricultural research data (ICAR, UAS Bangalore/Dharwad, KVK).
Uses pure Python standard library without heavy vector database dependencies.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


class AgriculturalRetrieverError(Exception):
    """Raised when knowledge retrieval or corpus loading fails."""
    pass


class AgriculturalRetriever:
    """
    Lightweight, deterministic lexical agricultural knowledge retriever.
    Scores knowledge entries based on crop alignment, domain keyword matches,
    topic overlap, and term frequency.
    """

    # Common agricultural crop synonyms for matching
    CROP_SYNONYMS: Dict[str, Set[str]] = {
        "paddy": {"paddy", "rice", "bhatta"},
        "ragi": {"ragi", "finger millet", "mandua"},
        "maize": {"maize", "corn", "makka"},
        "groundnut": {"groundnut", "peanut", "shenga"},
        "sugarcane": {"sugarcane", "cane", "kabbu"},
        "cotton": {"cotton", "kapas", "hatti"},
        "chilli": {"chilli", "chili", "mirchi", "menasinakayi"},
        "onion": {"onion", "pyaz", "eerulli"},
        "potato": {"potato", "aloo", "aalugadde"},
        "banana": {"banana", "plantain", "bale"},
        "tomato": {"tomato", "tamatar", "tometo"}
    }

    # Stopwords and generic non-specific terms ignored in content matching
    STOPWORDS: Set[str] = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
        "were", "will", "with", "my", "me", "i", "what", "should", "do", "how",
        "there", "been", "very", "little", "having", "trouble", "could", "some",
        "something", "eating", "seeing", "showing", "getting", "turning",
        "problem", "problems", "agricultural", "agriculture", "knowledge",
        "base", "covered", "issue", "issues", "farmer", "crop", "crops",
        "field", "plants", "plant", "help", "guide", "advice", "first", "check"
    }

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        top_k: int = 3,
        relevance_threshold: float = 1.0
    ):
        """
        Initialize the AgriculturalRetriever.

        Args:
            corpus_path: Path to agricultural_corpus.json. Defaults to project standard path.
            top_k: Number of top-scoring documents to return. Default is 3.
            relevance_threshold: Minimum score required for a document to be returned.
        """
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.corpus_path = corpus_path or self._get_default_corpus_path()
        self._corpus: List[Dict[str, Any]] = []
        self._load_corpus()

    def _get_default_corpus_path(self) -> str:
        """Resolves default location of agricultural_corpus.json."""
        project_root = Path(__file__).resolve().parent.parent.parent
        return str(project_root / "data" / "knowledge" / "agricultural_corpus.json")

    def _load_corpus(self) -> None:
        """Loads and validates the structured agricultural corpus."""
        if not os.path.exists(self.corpus_path):
            raise AgriculturalRetrieverError(f"Agricultural corpus not found at: {self.corpus_path}")

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise AgriculturalRetrieverError("Corpus format must be a list of JSON objects.")

            self._corpus = data
        except Exception as e:
            raise AgriculturalRetrieverError(f"Failed to read agricultural corpus: {str(e)}")

    @property
    def corpus_size(self) -> int:
        """Returns the number of loaded knowledge entries."""
        return len(self._corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Normalizes and tokenizes text into lowercase alphanumeric words."""
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens

    def _detect_query_crops(self, query_tokens: List[str], query_raw: str) -> Set[str]:
        """Detects crops explicitly mentioned in the query."""
        detected = set()
        query_lower = query_raw.lower()

        for canonical_crop, synonyms in self.CROP_SYNONYMS.items():
            for syn in synonyms:
                if syn in query_lower:
                    detected.add(canonical_crop)
                    break
        return detected

    def score_document(self, query: str, doc: Dict[str, Any]) -> float:
        """
        Calculates a deterministic relevance score between a query and a document.

        Scoring Components:
        - Crop Match (Weight: 3.5): Strong signal if the document matches the query's crop.
        - Keyword Match (Weight: 1.5): Matches between query words and document keywords.
        - Title Overlap (Weight: 1.0): Matches between query words and document title.
        - Topic Overlap (Weight: 0.8): Topic keywords present in query.
        - Content Overlap (Weight: 0.2 per unique word hit).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        query_set = set(query_tokens) - self.STOPWORDS
        if not query_set:
            return 0.0

        score = 0.0
        doc_crop = doc.get("crop", "").lower()
        detected_crops = self._detect_query_crops(query_tokens, query)

        # 1. Crop Match
        if doc_crop in detected_crops:
            score += 3.5
        elif detected_crops and doc_crop != "general":
            # If the user asked about a specific crop and doc is for another specific crop, penalize
            score -= 2.0

        # 2. Keyword Match
        doc_keywords = doc.get("keywords", [])
        for kw in doc_keywords:
            kw_lower = kw.lower()
            if kw_lower in query.lower():
                score += 1.5
            else:
                kw_tokens = set(self._tokenize(kw_lower)) - self.STOPWORDS
                overlap = len(query_set.intersection(kw_tokens))
                if overlap > 0:
                    score += 0.6 * overlap

        # 3. Title Overlap
        doc_title_tokens = set(self._tokenize(doc.get("title", ""))) - self.STOPWORDS
        title_overlap = len(query_set.intersection(doc_title_tokens))
        score += 1.0 * title_overlap

        # 4. Topic Overlap
        doc_topic = doc.get("topic", "").replace("_", " ")
        topic_tokens = set(self._tokenize(doc_topic)) - self.STOPWORDS
        topic_overlap = len(query_set.intersection(topic_tokens))
        score += 0.8 * topic_overlap

        # 5. Content Overlap
        doc_content_tokens = set(self._tokenize(doc.get("content", ""))) - self.STOPWORDS
        content_overlap = len(query_set.intersection(doc_content_tokens))
        score += 0.2 * min(content_overlap, 10)

        return max(0.0, score)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant knowledge entries for the query.

        Args:
            query: Farmer query in English.
            top_k: Optional override for number of entries. Defaults to self.top_k.

        Returns:
            List of dictionaries containing document metadata, content, source, and score.
            Returns empty list if no document exceeds relevance_threshold.
        """
        if not query or not query.strip() or not self._corpus:
            return []

        k = top_k if top_k is not None else self.top_k
        scored_docs = []

        for doc in self._corpus:
            score = self.score_document(query, doc)
            if score >= self.relevance_threshold:
                doc_copy = {
                    "id": doc.get("id"),
                    "crop": doc.get("crop"),
                    "topic": doc.get("topic"),
                    "title": doc.get("title"),
                    "content": doc.get("content"),
                    "source": doc.get("source"),
                    "score": round(score, 3)
                }
                scored_docs.append(doc_copy)

        # Sort descending by score, tie-break by ID for determinism
        scored_docs.sort(key=lambda d: (d["score"], d["id"]), reverse=True)
        return scored_docs[:k]

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved knowledge entries into a structured text block
        suitable for prompt injection.
        """
        if not retrieved_docs:
            return ""

        context_lines = ["--- RETRIEVED AGRICULTURAL KNOWLEDGE (ICAR/UAS) ---"]
        for idx, doc in enumerate(retrieved_docs, start=1):
            context_lines.append(f"[{idx}] Crop: {doc.get('crop', 'general').capitalize()} | Topic: {doc.get('topic', 'advisory')}")
            context_lines.append(f"Title: {doc.get('title', '')}")
            context_lines.append(f"Guidance: {doc.get('content', '')}")
            context_lines.append(f"Source: {doc.get('source', 'Agricultural Institute')}")
            context_lines.append("")

        return "\n".join(context_lines).strip()
