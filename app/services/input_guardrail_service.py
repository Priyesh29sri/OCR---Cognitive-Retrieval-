"""
Input Guardrail Service
Validates user input to prevent malicious queries, prompt injection, and inappropriate content
"""
from typing import Tuple, Optional
import re
from loguru import logger
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class InputGuardrailService:
    """Service for validating and filtering user input"""
    
    # Dangerous patterns that indicate prompt injection attempts
    PROMPT_INJECTION_PATTERNS = [
        r"ignore (previous|above|all) (instructions|prompts)",
        r"disregard (previous|above|all)",
        r"you are now|act as a|pretend (you are|to be)",
        r"system:\s*role",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[INST\]|\[/INST\]",
        r"forget (everything|your|all)",
        r"new instructions:",
        r"override",
    ]
    
    # PII patterns (basic detection)
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Input Guardrail Service initialized")
    
    def check_prompt_injection(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query contains prompt injection attempts
        
        Args:
            query: User's query text
            
        Returns:
            (is_safe, reason) - True if safe, False if malicious with reason
        """
        query_lower = query.lower()
        
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                reason = f"Potential prompt injection detected: {pattern}"
                logger.warning(f"Blocked query: {reason}")
                return False, reason
        
        return True, None
    
    def check_length(self, query: str, max_length: int = 2000) -> Tuple[bool, Optional[str]]:
        """
        Check if query length is within acceptable limits
        
        Args:
            query: User's query text
            max_length: Maximum allowed characters
            
        Returns:
            (is_valid, reason)
        """
        if len(query) > max_length:
            return False, f"Query too long ({len(query)} chars, max {max_length})"
        
        if len(query.strip()) < 3:
            return False, "Query too short (minimum 3 characters)"
        
        return True, None
    
    def check_toxicity(self, query: str) -> Tuple[bool, Optional[str], float]:
        """
        Use Gemini to detect toxic, hateful, or inappropriate content
        
        Args:
            query: User's query text
            
        Returns:
            (is_safe, reason, toxicity_score) - score from 0.0 (safe) to 1.0 (toxic)
        """
        try:
            prompt = f"""Analyze this user query for toxicity, hate speech, profanity, or inappropriate content.
Rate toxicity from 0.0 (completely safe) to 1.0 (highly toxic).

Query: "{query}"

Respond ONLY with a number between 0.0 and 1.0, nothing else."""
            
            response = self.model.generate_content(prompt)
            score_text = response.text.strip()
            
            # Extract number from response
            score = float(re.findall(r"0\.\d+|1\.0|0", score_text)[0])
            
            # Threshold: 0.6 or higher is considered toxic
            if score >= 0.6:
                return False, f"Content flagged as inappropriate (toxicity: {score})", score
            
            return True, None, score
            
        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            # Fail open (allow) to not block legitimate queries
            return True, None, 0.0
    
    def detect_pii(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect personally identifiable information (PII) in query
        
        Args:
            query: User's query text
            
        Returns:
            (has_pii, types_detected)
        """
        detected_types = []
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, query):
                detected_types.append(pii_type)
        
        if detected_types:
            return True, f"PII detected: {', '.join(detected_types)}"
        
        return False, None
    
    def validate(self, query: str, check_toxicity: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Full validation pipeline for user input
        
        Args:
            query: User's query text
            check_toxicity: Whether to run AI-based toxicity check (slower)
            
        Returns:
            (is_valid, reason) - True if valid, False with reason if invalid
        """
        # 1. Check length
        is_valid, reason = self.check_length(query)
        if not is_valid:
            return False, reason
        
        # 2. Check prompt injection
        is_safe, reason = self.check_prompt_injection(query)
        if not is_safe:
            return False, reason
        
        # 3. Detect PII (warn but don't block)
        has_pii, pii_info = self.detect_pii(query)
        if has_pii:
            logger.warning(f"Query contains PII: {pii_info}")
            # Don't block, just log
        
        # 4. Check toxicity (optional, slower)
        if check_toxicity:
            is_safe, reason, score = self.check_toxicity(query)
            if not is_safe:
                return False, reason
        
        return True, None


# Singleton instance
input_guardrail = InputGuardrailService()
