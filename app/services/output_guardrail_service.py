"""
Output Guardrail Service
Validates AI-generated output for hallucinations, safety, and quality
"""
from typing import Tuple, Optional, Dict
import re
from loguru import logger
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class OutputGuardrailService:
    """Service for validating AI-generated responses"""
    
    # Patterns indicating low-quality or evasive responses
    EVASIVE_PATTERNS = [
        r"i don't know",
        r"i cannot (answer|help|assist)",
        r"i'm not sure",
        r"as an ai",
        r"i don't have access",
        r"i'm unable to",
    ]
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Output Guardrail Service initialized")
    
    def check_hallucination(
        self, 
        answer: str, 
        evidence: list, 
        confidence_score: float
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if answer contains hallucinations using evidence verification
        
        Args:
            answer: Generated answer
            evidence: List of source text chunks
            confidence_score: Initial confidence score
            
        Returns:
            (is_faithful, reason, adjusted_confidence)
        """
        if not evidence or len(evidence) == 0:
            return True, "No evidence to verify against", confidence_score
        
        try:
            # Combine evidence
            evidence_text = "\n".join([str(e.get('text', e)) if isinstance(e, dict) else str(e) for e in evidence[:3]])
            
            prompt = f"""You are a fact-checker. Determine if the ANSWER is faithful to the EVIDENCE.

EVIDENCE:
{evidence_text}

ANSWER:
{answer}

Is the answer:
1. Supported by evidence (yes/no)
2. Contains made-up facts (yes/no)
3. Contradicts evidence (yes/no)

Respond ONLY with: FAITHFUL or HALLUCINATION"""
            
            response = self.model.generate_content(prompt)
            result = response.text.strip().upper()
            
            if "HALLUCINATION" in result:
                return False, "Answer contains unsupported claims", confidence_score * 0.5
            
            return True, None, confidence_score
            
        except Exception as e:
            logger.error(f"Hallucination check failed: {e}")
            return True, None, confidence_score
    
    def check_content_safety(self, answer: str) -> Tuple[bool, Optional[str]]:
        """
        Check if answer contains unsafe or inappropriate content
        
        Args:
            answer: Generated answer
            
        Returns:
            (is_safe, reason)
        """
        try:
            prompt = f"""Is this text safe, appropriate, and free from:
- Hate speech
- Violence
- Self-harm content  
- Illegal activities
- Adult content

Text: "{answer}"

Respond ONLY with: SAFE or UNSAFE"""
            
            response = self.model.generate_content(prompt)
            result = response.text.strip().upper()
            
            if "UNSAFE" in result:
                return False, "Answer contains unsafe content"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return True, None
    
    def check_quality(self, answer: str, min_length: int = 20) -> Tuple[bool, Optional[str]]:
        """
        Check if answer meets quality standards
        
        Args:
            answer: Generated answer
            min_length: Minimum acceptable length
            
        Returns:
            (is_quality, reason)
        """
        # Check length
        if len(answer.strip()) < min_length:
            return False, f"Answer too short ({len(answer)} chars, min {min_length})"
        
        # Check for evasive responses
        answer_lower = answer.lower()
        for pattern in self.EVASIVE_PATTERNS:
            if re.search(pattern, answer_lower):
                return False, f"Answer is evasive or non-committal"
        
        return True, None
    
    def detect_pii(self, answer: str) -> Tuple[bool, list]:
        """
        Detect PII that may have leaked into the response
        
        Args:
            answer: Generated answer
            
        Returns:
            (has_pii, detected_types)
        """
        detected = []
        
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        }
        
        for pii_type, pattern in patterns.items():
            if re.search(pattern, answer):
                detected.append(pii_type)
        
        return len(detected) > 0, detected
    
    def validate(
        self, 
        answer: str, 
        evidence: list = None, 
        confidence_score: float = 0.5,
        check_hallucination: bool = True
    ) -> Tuple[bool, Optional[str], float, Dict]:
        """
        Full validation pipeline for AI-generated output
        
        Args:
            answer: Generated answer
            evidence: Source evidence used
            confidence_score: Initial confidence
            check_hallucination: Whether to run hallucination check
            
        Returns:
            (is_valid, reason, adjusted_confidence, metadata)
        """
        metadata = {
            'original_confidence': confidence_score,
            'checks_passed': [],
            'checks_failed': [],
        }
        
        # 1. Check quality
        is_quality, reason = self.check_quality(answer)
        if not is_quality:
            metadata['checks_failed'].append('quality')
            return False, reason, confidence_score * 0.3, metadata
        metadata['checks_passed'].append('quality')
        
        # 2. Check content safety
        is_safe, reason = self.check_content_safety(answer)
        if not is_safe:
            metadata['checks_failed'].append('safety')
            return False, reason, 0.0, metadata
        metadata['checks_passed'].append('safety')
        
        # 3. Check hallucination
        if check_hallucination and evidence:
            is_faithful, reason, adjusted_conf = self.check_hallucination(
                answer, evidence, confidence_score
            )
            if not is_faithful:
                metadata['checks_failed'].append('hallucination')
                return False, reason, adjusted_conf, metadata
            confidence_score = adjusted_conf
            metadata['checks_passed'].append('hallucination')
        
        # 4. Detect PII (warn but don't block)
        has_pii, pii_types = self.detect_pii(answer)
        if has_pii:
            logger.warning(f"Answer contains PII: {pii_types}")
            metadata['pii_detected'] = pii_types
        
        metadata['final_confidence'] = confidence_score
        return True, None, confidence_score, metadata


# Singleton instance
output_guardrail = OutputGuardrailService()
