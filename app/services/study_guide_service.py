"""
Study Guide Generator for ICDI-X
==================================
Novel feature: Converts any uploaded document into a Bloom's Taxonomy study guide.

Uniqueness (paper Section 4.3):
- Perplexity: no study-guide feature
- NotebookLM: has "suggested questions" but NOT Bloom's taxonomy classification
- ICDI-X: full 6-level Bloom's taxonomy (Remember → Create) + concept map + vocabulary

Bloom's Taxonomy levels (Anderson & Krathwohl, 2001):
  L1 Remember   — Recall facts
  L2 Understand — Explain concepts
  L3 Apply      — Use in new situations
  L4 Analyze    — Draw connections
  L5 Evaluate   — Justify decisions
  L6 Create     — Produce new work

Use cases:
  - Students: chapter → full study guide with progressive questions
  - Researchers: paper → critical analysis Q&A set
  - Professionals: report → discussion prompts
"""

import os
import json
from typing import List, Dict, Optional
from loguru import logger
from google import genai
from together import Together


BLOOMS_LEVELS = {
    "remember":   "Recall facts and basic concepts from the document",
    "understand": "Explain ideas or concepts in your own words",
    "apply":      "Use information from the document in new situations",
    "analyze":    "Draw connections, examine assumptions, compare elements",
    "evaluate":   "Justify decisions or critique approaches described",
    "create":     "Design, propose, or produce something new based on the document",
}


class StudyGuideService:
    """
    Bloom's Taxonomy study guide generator.

    Pipeline:
    1. Take top-K representative chunks from the document (via RAG retrieval)
    2. Prompt LLM to generate all 6 Bloom's levels of questions
    3. Extract vocabulary, key concepts, and concept map
    4. Return structured JSON ready for frontend rendering
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None
        logger.info("✅ Study Guide Service initialized")

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
                logger.warning(f"Gemini failed in StudyGuideService: {e}")

        if self.together_client:
            try:
                resp = self.together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2800,
                    temperature=0.3,
                )
                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"Together AI failed in StudyGuideService: {e}")
        return ""

    # ─────────────────────────────────────────────────────────────
    # Main public method
    # ─────────────────────────────────────────────────────────────

    async def generate_study_guide(
        self,
        doc_id: str,
        chunks: List[Dict],           # [{"text": str, "score": float}]
        filename: str = "document",
    ) -> Dict:
        """
        Generate a full Bloom's Taxonomy study guide from document chunks.

        Returns
        -------
        {
            "doc_id": str,
            "title": str,
            "summary": str,
            "key_concepts": [str, ...],
            "vocabulary": [{"term": str, "definition": str}, ...],
            "blooms_questions": {
                "remember":   [str, ...],
                "understand": [str, ...],
                "apply":      [str, ...],
                "analyze":    [str, ...],
                "evaluate":   [str, ...],
                "create":     [str, ...],
            },
            "concept_map": [{"from": str, "relation": str, "to": str}, ...],
            "estimated_study_time_minutes": int,
            "blooms_taxonomy_info": {level: description, ...},
        }
        """
        if not chunks:
            return self._empty_response(doc_id, filename)

        text = "\n\n".join(c.get("text", "") for c in chunks[:12])[:5500]

        prompt = f"""You are an expert educator and instructional designer specialising in Bloom's Taxonomy.
Create a comprehensive study guide from the following document content.

DOCUMENT: {filename}
CONTENT:
{text}

Respond with ONLY valid JSON — no markdown, no extra text:
{{
  "title": "Study Guide: [document topic in 5-8 words]",
  "summary": "3-4 sentence executive summary of the document's main purpose and findings.",
  "key_concepts": [
    "concept1 (specific to this document)",
    "concept2",
    "concept3",
    "concept4",
    "concept5"
  ],
  "vocabulary": [
    {{"term": "technical or domain-specific term", "definition": "plain-language definition"}},
    {{"term": "another term", "definition": "definition"}}
  ],
  "blooms_questions": {{
    "remember": [
      "What is [specific fact from document]?",
      "List the [specific items] mentioned."
    ],
    "understand": [
      "Explain in your own words what [concept] means.",
      "Describe the relationship between [X] and [Y] as presented."
    ],
    "apply": [
      "How would you use [method from document] to solve [problem type]?",
      "Apply [principle] to a real-world scenario outside the document."
    ],
    "analyze": [
      "Compare and contrast [A] and [B] from the document.",
      "What assumptions underlie [argument/method] described?"
    ],
    "evaluate": [
      "What are the strengths and limitations of [approach] described?",
      "How well does [method/system] address the stated problem? Justify."
    ],
    "create": [
      "Design an experiment to test [hypothesis from document].",
      "Propose an improvement to [method/system] based on the document's findings."
    ]
  }},
  "concept_map": [
    {{"from": "concept_a", "relation": "leads_to", "to": "concept_b"}},
    {{"from": "concept_b", "relation": "supports",  "to": "concept_c"}},
    {{"from": "concept_c", "relation": "enables",   "to": "concept_d"}}
  ],
  "estimated_study_time_minutes": 30
}}

Critical requirements:
- ALL questions must reference SPECIFIC content from the document (not generic)
- vocabulary should have at least 4 entries
- concept_map should have at least 3 relationships
- Use the exact Bloom's level names as JSON keys"""

        raw = self._call_llm(prompt)

        try:
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(raw)
        except Exception as e:
            logger.error(f"StudyGuideService JSON parse error: {e} | raw[:300]={raw[:300]}")
            result = self._fallback_guide(filename)

        result["doc_id"] = doc_id
        result["blooms_taxonomy_info"] = BLOOMS_LEVELS
        return result

    # ─────────────────────────────────────────────────────────────
    # Fallbacks
    # ─────────────────────────────────────────────────────────────

    def _fallback_guide(self, filename: str) -> Dict:
        return {
            "title": f"Study Guide: {filename}",
            "summary": "Document processed successfully. Study guide generation encountered a formatting issue.",
            "key_concepts": [],
            "vocabulary": [],
            "blooms_questions": {level: [] for level in BLOOMS_LEVELS},
            "concept_map": [],
            "estimated_study_time_minutes": 30,
            "error": "Study guide generation encountered an issue. Please retry.",
        }

    def _empty_response(self, doc_id: str, filename: str) -> Dict:
        result = self._fallback_guide(filename)
        result["doc_id"] = doc_id
        result["blooms_taxonomy_info"] = BLOOMS_LEVELS
        result["error"] = "No content available. Document may still be indexing — wait 15 s and retry."
        return result
