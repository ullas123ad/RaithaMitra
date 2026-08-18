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

from model.advisory.crop_identifier import (
    CROP_CANONICAL_MAP,
    SUPPORTED_CROPS,
    detect_crop_from_text,
    normalize_crop_name,
)


class AgriculturalRetrieverError(Exception):
    """Raised when knowledge retrieval or corpus loading fails."""
    pass


class AgriculturalRetriever:
    """
    Lightweight, deterministic lexical agricultural knowledge retriever.
    Scores knowledge entries based on crop alignment, domain symptom keywords,
    topic overlap, and term frequency with precision filtering and strict
    cross-crop contamination prevention.
    """

    # Common agricultural crop synonyms for matching (English & Kannada aliases)
    CROP_SYNONYMS: Dict[str, Set[str]] = {
        "paddy": {"paddy", "rice", "bhatta", "ಭತ್ತ"},
        "ragi": {"ragi", "finger millet", "mandua", "fingermillet", "ರಾಗಿ"},
        "maize": {"maize", "corn", "makka", "mekkejola", "ಮೆಕ್ಕೆಜೋಳ"},
        "groundnut": {"groundnut", "peanut", "shenga", "kadlekai", "ಕಡಲೆಕಾಯಿ"},
        "sugarcane": {"sugarcane", "cane", "kabbu", "ಕಬ್ಬು"},
        "cotton": {"cotton", "kapas", "hatti", "ಹತ್ತಿ"},
        "chilli": {"chilli", "chili", "mirchi", "menasinakai", "menasinakayi", "capsicum", "ಮೆಣಸಿನಕಾಯಿ"},
        "onion": {"onion", "pyaz", "eerulli", "ಈರುಳ್ಳಿ"},
        "potato": {"potato", "aloo", "aalugadde", "ಆಲೂಗಡ್ಡೆ"},
        "banana": {"banana", "plantain", "bale", "baale", "ಬಾಳೆ"},
        "tomato": {"tomato", "tamatar", "tometo", "tamota", "ಟೊಮ್ಯಾಟೊ", "ಟೊಮೆಟೊ"}
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
        "field", "plants", "plant", "help", "guide", "advice", "first", "check",
        "also", "few", "day", "days", "now", "properly", "past"
    }

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        top_k: int = 3,
        relevance_threshold: float = 1.5,
        score_gap_ratio: float = 0.5
    ):
        """
        Initialize the AgriculturalRetriever.

        Args:
            corpus_path: Path to agricultural_corpus.json. Defaults to project standard path.
            top_k: Maximum number of top-scoring documents to return. Default is 3.
            relevance_threshold: Minimum score required for a document to be returned.
            score_gap_ratio: Minimum ratio of top document's score required for secondary documents.
        """
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.score_gap_ratio = score_gap_ratio
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

    @staticmethod
    def _stem(token: str) -> str:
        """Lightweight rule-based English suffix normalizer."""
        t = token.lower().strip()
        if not t:
            return ""

        # Common agricultural synonyms / irregulars
        irregulars = {
            "leaves": "leaf",
            "leaf": "leaf",
            "dried": "dry",
            "drying": "dry",
            "dry": "dry",
            "dries": "dry",
            "curled": "curl",
            "curling": "curl",
            "curls": "curl",
            "curl": "curl",
            "rained": "rain",
            "raining": "rain",
            "rains": "rain",
            "rainfall": "rain",
            "rain": "rain",
            "yellowing": "yellow",
            "yellowed": "yellow",
            "yellow": "yellow",
            "wilting": "wilt",
            "wilted": "wilt",
            "wilt": "wilt",
            "waterlogging": "waterlog",
            "waterlogged": "waterlog",
            "spots": "spot",
            "spotting": "spot",
            "spotted": "spot",
            "spot": "spot",
            "holes": "hole",
            "hole": "hole",
            "borers": "borer",
            "borer": "borer",
            "insects": "insect",
            "insect": "insect",
            "pests": "pest",
            "pest": "pest",
            "diseases": "disease",
            "disease": "disease",
            "rots": "rot",
            "rotting": "rot",
            "rotted": "rot",
            "rot": "rot",
        }
        if t in irregulars:
            return irregulars[t]

        # Standard rule-based suffixes
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("s") and not t.endswith("ss") and len(t) > 2:
            t = t[:-1]

        if t.endswith("ing") and len(t) > 4:
            t = t[:-3]
            if len(t) > 2 and t[-1] == t[-2]:
                t = t[:-1]
        elif t.endswith("ed") and len(t) > 3:
            t = t[:-2]
            if len(t) > 2 and t[-1] == t[-2]:
                t = t[:-1]

        return irregulars.get(t, t)

    def _tokenize(self, text: str) -> List[str]:
        """Normalizes, tokenizes, and stems text into lowercase alphanumeric words."""
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [t.strip() for t in cleaned.split() if t.strip()]
        return tokens

    def _get_stemmed_set(self, text: str) -> Set[str]:
        """Tokenizes text and returns a set of stemmed non-stopword tokens."""
        tokens = self._tokenize(text)
        return {self._stem(t) for t in tokens if t not in self.STOPWORDS and self._stem(t) not in self.STOPWORDS}

    def _detect_query_crops(self, query_raw: str) -> Set[str]:
        """Detects crops mentioned in the query string."""
        detected = set()
        canonical = detect_crop_from_text(query_raw)
        if canonical:
            detected.add(canonical)
            return detected

        query_lower = query_raw.lower()
        for canon, synonyms in self.CROP_SYNONYMS.items():
            for syn in synonyms:
                if syn in query_lower:
                    detected.add(canon)
                    break
        return detected

    def score_document(
        self,
        query: str,
        doc: Dict[str, Any],
        target_crop: Optional[str] = None
    ) -> float:
        """
        Calculates a deterministic relevance score between a query and a document.

        Scoring & Safety Rules:
        1. Target Crop Strictness:
           - If a target crop is specified or detected, documents for a DIFFERENT specific crop
             are disqualified immediately (score = 0.0) to prevent cross-crop contamination.
           - Documents matching the target crop receive a +3.0 alignment bonus.
           - General agronomic documents ('general') remain eligible without crop bonus.
        2. Symptom & Problem Keyword Match (Weight: 2.0 / 0.8): Matches domain symptoms and remedies.
        3. Title Overlap (Weight: 1.2): Matches between query terms and document title.
        4. Topic Overlap (Weight: 1.0): Matches between query terms and topic category.
        5. Content Overlap (Weight: 0.2 per unique word hit, max 2.0).
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        query_stemmed_set = self._get_stemmed_set(query)
        if not query_stemmed_set:
            return 0.0

        doc_crop = doc.get("crop", "").lower()
        query_lower = query.lower()

        # Determine effective target crop
        effective_crop = normalize_crop_name(target_crop) if target_crop else None
        if not effective_crop:
            detected = self._detect_query_crops(query)
            if len(detected) == 1:
                effective_crop = next(iter(detected))

        all_crop_names = set().union(*self.CROP_SYNONYMS.values())
        all_crop_stemmed = {self._stem(c) for c in all_crop_names}

        score = 0.0
        has_crop_match = False

        # 1. Crop Match & Strict Cross-Crop Exclusion
        if effective_crop:
            if doc_crop == effective_crop:
                score += 3.0
                has_crop_match = True
            elif doc_crop == "general":
                # General agronomy docs eligible on symptom merits
                has_crop_match = False
            else:
                # STRICT REJECTION: Document is for a different specific crop
                return 0.0
        else:
            detected_crops = self._detect_query_crops(query)
            if detected_crops:
                if doc_crop in detected_crops:
                    score += 3.0
                    has_crop_match = True
                elif doc_crop != "general":
                    return 0.0

        # 2. Symptom & Domain Keyword Match (excluding pure crop name repetition)
        doc_keywords = doc.get("keywords", [])
        symptom_kw_hits = 0

        for kw in doc_keywords:
            kw_lower = kw.lower()
            if kw_lower in all_crop_names:
                continue

            if kw_lower in query_lower:
                score += 2.0
                symptom_kw_hits += 1
            else:
                kw_stemmed = {self._stem(t) for t in self._tokenize(kw_lower)} - self.STOPWORDS - all_crop_stemmed
                overlap = len(query_stemmed_set.intersection(kw_stemmed))
                if overlap > 0:
                    score += 0.8 * overlap
                    symptom_kw_hits += 1

        # 3. Title Overlap
        doc_title_stemmed = {self._stem(t) for t in self._tokenize(doc.get("title", ""))} - self.STOPWORDS - all_crop_stemmed
        title_overlap = len(query_stemmed_set.intersection(doc_title_stemmed))
        score += 1.2 * title_overlap

        # 4. Topic Overlap
        doc_topic = doc.get("topic", "").replace("_", " ")
        topic_stemmed = {self._stem(t) for t in self._tokenize(doc_topic)} - self.STOPWORDS
        topic_overlap = len(query_stemmed_set.intersection(topic_stemmed))
        score += 1.0 * topic_overlap

        # 5. Content Overlap
        doc_content_stemmed = {self._stem(t) for t in self._tokenize(doc.get("content", ""))} - self.STOPWORDS - all_crop_stemmed
        content_overlap = len(query_stemmed_set.intersection(doc_content_stemmed))
        score += 0.2 * min(content_overlap, 10)

        # Policy: If crop matched, but there is zero symptom, title, or topic overlap, penalize
        if has_crop_match and (symptom_kw_hits == 0 and title_overlap == 0 and topic_overlap == 0):
            score *= 0.3

        return max(0.0, score)

    def retrieve(
        self,
        query: str,
        crop: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant knowledge entries for the query with precision filtering.

        Args:
            query: Farmer query in English (or Kannada).
            crop: Optional canonical crop name to ground retrieval context.
            top_k: Optional override for number of entries. Defaults to self.top_k.

        Returns:
            List of dictionaries containing document metadata, content, source, and score.
            Returns empty list if no document exceeds relevance_threshold.
        """
        if not query or not query.strip() or not self._corpus:
            return []

        k = top_k if top_k is not None else self.top_k
        scored_docs = []
        norm_crop = normalize_crop_name(crop) if crop else None

        for doc in self._corpus:
            score = self.score_document(query, doc, target_crop=norm_crop)
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

        # Relative score gap filter: keep secondary docs only if they are competitive with top doc
        if scored_docs:
            top_score = scored_docs[0]["score"]
            scored_docs = [
                d for d in scored_docs
                if d["score"] >= (self.score_gap_ratio * top_score)
            ]

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
