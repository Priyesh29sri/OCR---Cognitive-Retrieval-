"""
Cross-Document Contradiction Detector for ICDI-X
=================================================
Novel feature: Automatically detects contradictions and agreements ACROSS documents.

Uniqueness (paper Section 4.2):
- Perplexity: single-source web answers, no cross-document comparison
- NotebookLM: multi-doc Q&A but NO contradiction detection
- ICDI-X: structured contradiction taxonomy with severity + resolution suggestions

Contradiction Taxonomy:
  • direct      — factual opposition ("X causes Y" vs "X does NOT cause Y")
  • implied     — incompatible methodological assumptions
  • scope       — same claim but different scope/conditions
  • methodology — different methods lead to conflicting conclusions

Use cases:
  - Research: compare two papers on the same hypothesis
  - Legal: detect where a contract conflicts with a regulation
  - Medical: flag conflicting clinical guidelines
"""

import os
import json
from typing import List, Dict, Optional
from loguru import logger
from google import genai
from together import Together


class ContradictionDetectorService:
    """
    Cross-document contradiction detection.

    Architecture:
    1. Accept top-K chunks from two documents (already retrieved by RAG)
    2. Align chunks by topic using prompt-based semantic clustering
    3. LLM identifies direct/implied/scope/methodology contradictions
    4. Returns structured report with severity, confidence, resolution
    """

    CONTRADICTION_TYPES = {
        "direct": "Factual opposition (Doc A states X; Doc B states NOT-X)",
        "implied": "Incompatible underlying assumptions",
        "scope": "Same claim but different conditions/populations/scales",
        "methodology": "Different methods lead to conflicting conclusions",
    }

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None
        logger.info("✅ Contradiction Detector Service initialized")

    # ─────────────────────────────────────────────────────────────
    # LLM wrapper
    # ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self.gemini_client:
            try:
                resp = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                return resp.text
            except Exception as e:
                logger.warning(f"Gemini failed in ContradictionDetector: {e}")

        if self.together_client:
            try:
                resp = self.together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2200,
                    temperature=0.15,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Together AI failed in ContradictionDetector: {e}")
        return ""

    # ─────────────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────────────

    async def detect_contradictions(
        self,
        doc_a_chunks: List[str],
        doc_b_chunks: List[str],
        doc_a_name: str = "Document A",
        doc_b_name: str = "Document B",
        topic: str = "",
    ) -> Dict:
        """
        Detect contradictions between two document chunk sets.

        Parameters
        ----------
        doc_a_chunks / doc_b_chunks : list of text strings (top-K retrieved chunks)
        doc_a_name / doc_b_name     : human-readable document names
        topic                       : optional focus topic (empty = analyse all)

        Returns
        -------
        {
            "contradictions": [
                {
                    "topic": str,
                    "doc_a_claim": str,
                    "doc_b_claim": str,
                    "contradiction_type": "direct|implied|scope|methodology",
                    "severity": "high|medium|low",
                    "confidence": float,      # 0-1
                    "resolution": str,
                }
            ],
            "agreements": [str, ...],
            "summary": str,
            "overall_agreement_score": float,  # 0 (total conflict) – 1 (total agreement)
            "doc_a_name": str,
            "doc_b_name": str,
        }
        """
        if not doc_a_chunks or not doc_b_chunks:
            return self._empty_response(doc_a_name, doc_b_name)

        text_a = "\n\n".join(doc_a_chunks[:6])[:3000]
        text_b = "\n\n".join(doc_b_chunks[:6])[:3000]
        focus = f"Focus specifically on the topic: '{topic}'" if topic else "Analyse all claims present in both documents."

        prompt = f"""You are an expert scientific fact-checker and logical analyst.

Compare these two document excerpts and produce a structured contradiction / agreement report.

=== {doc_a_name} ===
{text_a}

=== {doc_b_name} ===
{text_b}

{focus}

Respond with ONLY valid JSON — no markdown fences, no extra text:
{{
  "contradictions": [
    {{
      "topic": "specific topic being compared",
      "doc_a_claim": "exact or paraphrased claim from {doc_a_name}",
      "doc_b_claim": "exact or paraphrased claim from {doc_b_name}",
      "contradiction_type": "direct|implied|scope|methodology",
      "severity": "high|medium|low",
      "confidence": 0.85,
      "resolution": "One sentence explaining how to reconcile, or N/A"
    }}
  ],
  "agreements": [
    "Claim or finding that both documents share",
    "Another shared point"
  ],
  "summary": "2-3 sentence synthesis of overall agreement/conflict.",
  "overall_agreement_score": 0.6
}}

Contradiction types:
- direct      : {self.CONTRADICTION_TYPES["direct"]}
- implied     : {self.CONTRADICTION_TYPES["implied"]}
- scope       : {self.CONTRADICTION_TYPES["scope"]}
- methodology : {self.CONTRADICTION_TYPES["methodology"]}

Severity guide:
- high   : Core factual claims directly conflict
- medium : Partial or conditional conflict
- low    : Minor differences in emphasis or framing

If no contradictions exist, return an empty "contradictions" list and a high overall_agreement_score."""

        raw = self._call_llm(prompt)

        try:
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
        except Exception as e:
            logger.error(f"ContradictionDetector JSON parse error: {e} | raw[:300]={raw[:300]}")
            result = {
                "contradictions": [],
                "agreements": [],
                "summary": "Comparison completed — structured output unavailable.",
                "overall_agreement_score": 0.5,
                "parse_error": str(e),
            }

        result["doc_a_name"] = doc_a_name
        result["doc_b_name"] = doc_b_name
        result["contradiction_taxonomy"] = self.CONTRADICTION_TYPES
        return result

    # ─────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────

    def _empty_response(self, doc_a_name: str, doc_b_name: str) -> Dict:
        return {
            "contradictions": [],
            "agreements": [],
            "summary": "No content retrieved for one or both documents.",
            "overall_agreement_score": 0.5,
            "doc_a_name": doc_a_name,
            "doc_b_name": doc_b_name,
            "error": "One or both document IDs returned no chunks. Ensure documents are indexed.",
        }
