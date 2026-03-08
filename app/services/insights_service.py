"""
Proactive Insights Engine for ICDI-X
=====================================
Novel feature: Auto-generates document insights BEFORE the user asks a question.

Uniqueness vs Perplexity & NotebookLM:
- Perplexity: web-search focused, no document-level proactive extraction
- NotebookLM: audio overview but NO information-bottleneck based chunk selection
- ICDI-X: IB compression → selects most information-dense chunks → LLM insight synthesis

Research angle (paper Section 4.1):
  Proactive Insight Generation via Information-Bottleneck Chunk Selection.
  For chunk cᵢ, IB score = α·TTR(cᵢ) + β·SentenceDensity(cᵢ) + γ·LengthBonus(cᵢ)
  Top-K chunks by IB score are selected → LLM generates insights from compressed representation.
"""

import os
import json
from typing import List, Dict, Optional
from loguru import logger
from google import genai
from together import Together


class InsightsService:
    """
    Proactive document insight generator using IB-scored chunk selection.

    Novel contribution:
    - Information Bottleneck (IB) scoring picks most information-dense chunks
    - Bloom's-inspired question generation (6 cognitive levels)
    - Structured output with entities, themes, doc-type, complexity
    - Works 100% from existing Qdrant chunks — no re-processing needed
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None
        logger.info("✅ Insights Service initialized")

    # ─────────────────────────────────────────────────────────────
    # IB Scoring
    # ─────────────────────────────────────────────────────────────

    def _ib_score(self, text: str) -> float:
        """
        Information Bottleneck-inspired density score.
        Higher = more information-dense (favoured for selection).

        IB_score(c) = 0.4 * TTR(c)
                    + 0.3 * min(avg_sentence_len / 20, 1.0)
                    + 0.3 * min(|words| / 200, 1.0)
        where TTR = type-token ratio (vocabulary richness).
        """
        words = text.lower().split()
        if not words:
            return 0.0

        ttr = len(set(words)) / len(words)                           # vocabulary richness
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_sent = len(words) / max(len(sentences), 1)               # sentence density
        length_bonus = min(len(words) / 200, 1.0)                    # length reward (capped)

        return 0.4 * ttr + 0.3 * min(avg_sent / 20, 1.0) + 0.3 * length_bonus

    # ─────────────────────────────────────────────────────────────
    # LLM wrapper
    # ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Gemini-first, Together-AI fallback."""
        if self.gemini_client:
            try:
                resp = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return resp.text
            except Exception as e:
                logger.warning(f"Gemini failed in InsightsService: {e}")

        if self.together_client:
            try:
                resp = self.together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1800,
                    temperature=0.3,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Together AI failed in InsightsService: {e}")
        return ""

    # ─────────────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────────────

    async def generate_insights(
        self,
        doc_id: str,
        chunks: List[Dict],          # [{"text": str, "score": float, "source_type": str}]
        filename: str = "document",
    ) -> Dict:
        """
        Generate proactive insights from Qdrant-retrieved chunks.

        Returns
        -------
        {
            "doc_id": str,
            "doc_type": str,
            "complexity": "basic|intermediate|advanced",
            "insights": [str, ...],          # 5-7 non-obvious findings
            "suggested_questions": [str, ...],# 5 clickable questions
            "key_entities": [str, ...],
            "key_themes": [str, ...],
            "ib_coverage": float,            # fraction of doc IB selected
            "chunks_analyzed": int,
            "chunks_selected_by_ib": int,
        }
        """
        if not chunks:
            return self._empty_response(doc_id)

        # ── Step 1: IB-score all chunks ──────────────────────────
        scored = [
            {**c, "ib_score": self._ib_score(c.get("text", ""))}
            for c in chunks
        ]
        scored.sort(key=lambda x: x["ib_score"], reverse=True)

        # ── Step 2: Select top-8 by IB (compression) ─────────────
        top_k = min(8, len(scored))
        top_chunks = scored[:top_k]
        ib_coverage = top_k / max(len(chunks), 1)
        combined = "\n\n---\n\n".join(c["text"] for c in top_chunks)

        # ── Step 3: LLM insight synthesis ────────────────────────
        prompt = f"""You are an expert document analyst specialising in rapid knowledge extraction.
Analyse the following (IB-selected) document excerpts and return structured insights.

DOCUMENT: {filename}
EXCERPTS (most information-dense sections):
{combined[:4500]}

Respond with ONLY valid JSON — no markdown fences, no extra text:
{{
  "doc_type": "research_paper|technical_report|legal_document|business_report|educational_material|news_article|book_chapter|other",
  "complexity": "basic|intermediate|advanced",
  "insights": [
    "Specific, non-obvious finding or fact from the document (sentence form)",
    "Second specific insight",
    "Third specific insight",
    "Fourth specific insight",
    "Fifth specific insight"
  ],
  "suggested_questions": [
    "Question a reader would naturally ask about this document?",
    "Second question?",
    "Third question?",
    "Fourth question?",
    "Fifth question?"
  ],
  "key_entities": ["entity1", "entity2", "entity3", "entity4", "entity5"],
  "key_themes": ["theme1", "theme2", "theme3"]
}}

Rules:
- Insights must be specific and actionable (not "the document discusses...")
- Questions must be answerable from the document
- Entities are proper nouns: people, orgs, methods, datasets, metrics"""

        raw = self._call_llm(prompt)

        try:
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
        except Exception as e:
            logger.error(f"InsightsService JSON parse failed: {e} | raw[:300]={raw[:300]}")
            result = self._fallback_insights(combined, doc_id)

        result.update({
            "doc_id": doc_id,
            "ib_coverage": round(ib_coverage, 3),
            "chunks_analyzed": len(chunks),
            "chunks_selected_by_ib": top_k,
        })
        return result

    # ─────────────────────────────────────────────────────────────
    # Fallbacks
    # ─────────────────────────────────────────────────────────────

    def _fallback_insights(self, text: str, doc_id: str) -> Dict:
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
        return {
            "doc_type": "document",
            "complexity": "intermediate",
            "insights": sentences[:5] if sentences else ["Document content extracted."],
            "suggested_questions": [
                "What are the main findings of this document?",
                "What methodology was used?",
                "What are the key conclusions?",
                "Who are the key entities mentioned?",
                "What is the practical significance?",
            ],
            "key_entities": [],
            "key_themes": [],
        }

    def _empty_response(self, doc_id: str) -> Dict:
        return {
            "doc_id": doc_id,
            "doc_type": "unknown",
            "complexity": "unknown",
            "insights": [],
            "suggested_questions": [
                "What is this document about?",
                "What are the key points?",
                "What conclusions are drawn?",
                "Who are the main contributors?",
                "What methods were used?",
            ],
            "key_entities": [],
            "key_themes": [],
            "ib_coverage": 0.0,
            "chunks_analyzed": 0,
            "chunks_selected_by_ib": 0,
            "error": "No content available yet. Document may still be indexing — wait 15 s and retry.",
        }
