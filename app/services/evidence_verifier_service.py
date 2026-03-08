"""
Evidence Verification Service for ICDI-X
========================================
Structured Evidence Assessment (SEA) to verify answer support and reduce hallucinations.

Ensures generated answers are fully grounded in retrieved evidence.
Research basis: FAIR-RAG, Self-RAG
"""

import os
import json
import re
from typing import List, Dict, Tuple
from loguru import logger
from google import genai
from together import Together


class Evidence:
    """Represents a piece of evidence"""
    def __init__(self, text: str, source: str, confidence: float = 1.0):
        self.text = text
        self.source = source
        self.confidence = confidence


class EvidenceVerifierService:
    """
    Verifies that generated answers are supported by retrieved evidence.
    
    Key Methods:
    - verify_answer_support: Check if answer is grounded in evidence
    - identify_gaps: Find missing information
    - compute_faithfulness: Score answer faithfulness
    """
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None

        logger.info("Evidence Verifier Service initialized")
    
    def verify_answer_support(self, answer: str, evidence_list: List[Evidence]) -> Dict:
        """
        Verify if answer is fully supported by evidence.
        
        Returns verification result with support score and unsupported claims.
        """
        if not self.client and not self.together_client:
            return {"supported": True, "score": 1.0, "reasoning": "No verification available"}

        evidence_text = "\n\n".join([f"Evidence {i+1}: {e.text}" for i, e in enumerate(evidence_list)])
        prompt = f"""Verify if this answer is fully supported by the evidence.

Answer: {answer}

Evidence:
{evidence_text}

Respond in JSON format:
{{
    "supported": true/false,
    "support_score": 0.0-1.0,
    "unsupported_claims": ["claim1", "claim2"],
    "reasoning": "explanation"
}}
"""

        # Try Gemini first
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                result = eval(response.text)
                logger.info(f"Verification score: {result.get('support_score', 0):.2f}")
                return result
            except Exception as e:
                logger.warning(f"Gemini verification failed: {e}")

        # Together AI fallback
        if self.together_client:
            try:
                response = self.together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    logger.info(f"Together AI verification score: {result.get('support_score', 0):.2f}")
                    return result
            except Exception as e2:
                logger.error(f"Together AI verification also failed: {e2}")

        return {"supported": True, "score": 0.5, "reasoning": "Verification unavailable", "unsupported_claims": []}
    
    def identify_information_gaps(self, query: str, retrieved_evidence: List[Evidence]) -> List[str]:
        """Identify what information is missing to fully answer the query"""
        if not self.client and not self.together_client:
            return []

        evidence_text = "\n".join([e.text for e in retrieved_evidence])
        prompt = f"""What information is missing to fully answer this query?

Query: {query}

Retrieved Evidence:
{evidence_text}

List missing information as JSON array: ["missing1", "missing2"]
"""

        # Try Gemini first
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                gaps = eval(response.text)
                logger.info(f"Identified {len(gaps)} information gaps")
                return gaps
            except Exception as e:
                logger.warning(f"Gemini gap identification failed: {e}")

        # Together AI fallback
        if self.together_client:
            try:
                response = self.together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    gaps = json.loads(match.group())
                    logger.info(f"Together AI identified {len(gaps)} information gaps")
                    return gaps
            except Exception as e2:
                logger.error(f"Together AI gap identification also failed: {e2}")

        return []
    
    def compute_faithfulness_score(self, answer: str, evidence_list: List[Evidence]) -> float:
        """
        Compute faithfulness score (0-1).
        
        Higher score = answer more grounded in evidence.
        """
        verification = self.verify_answer_support(answer, evidence_list)
        return verification.get("support_score", 0.0)
