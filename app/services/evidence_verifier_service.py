"""
Evidence Verification Service for ICDI-X
========================================
Structured Evidence Assessment (SEA) to verify answer support and reduce hallucinations.

Ensures generated answers are fully grounded in retrieved evidence.
Research basis: FAIR-RAG, Self-RAG
"""

import os
from typing import List, Dict, Tuple
from loguru import logger
from google import genai


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
        logger.info("Evidence Verifier Service initialized")
    
    def verify_answer_support(self, answer: str, evidence_list: List[Evidence]) -> Dict:
        """
        Verify if answer is fully supported by evidence.
        
        Returns verification result with support score and unsupported claims.
        """
        if not self.client:
            return {"supported": True, "score": 1.0, "reasoning": "No verification available"}
        
        try:
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
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            result = eval(response.text)
            logger.info(f"Verification score: {result.get('support_score', 0):.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"supported": False, "score": 0.0, "reasoning": str(e)}
    
    def identify_information_gaps(self, query: str, retrieved_evidence: List[Evidence]) -> List[str]:
        """Identify what information is missing to fully answer the query"""
        if not self.client:
            return []
        
        try:
            evidence_text = "\n".join([e.text for e in retrieved_evidence])
            
            prompt = f"""What information is missing to fully answer this query?

Query: {query}

Retrieved Evidence:
{evidence_text}

List missing information as JSON array: ["missing1", "missing2"]
"""
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            
            gaps = eval(response.text)
            logger.info(f"Identified {len(gaps)} information gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            return []
    
    def compute_faithfulness_score(self, answer: str, evidence_list: List[Evidence]) -> float:
        """
        Compute faithfulness score (0-1).
        
        Higher score = answer more grounded in evidence.
        """
        verification = self.verify_answer_support(answer, evidence_list)
        return verification.get("support_score", 0.0)
