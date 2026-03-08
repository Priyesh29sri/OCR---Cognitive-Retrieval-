"""
Information Bottleneck Filter Service for ICDI-X
================================================
Compresses retrieved context using Information Theory principles.

Filters noise and retains only answer-relevant information by maximizing
mutual information with the answer while minimizing mutual information with
the original retrieved passage.

Research basis: Information Bottleneck Theory, IB-RAG
"""

import os
import json
import re
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
from google import genai
from together import Together


class InformationBottleneckService:
    """
    Information Bottleneck-based context compression.
    
    Mathematical Foundation:
    Maximize: I(Z;Y) - β·I(Z;X)
    
    Where:
    - X = retrieved context
    - Y = target answer
    - Z = compressed representation
    - β = compression parameter
    
    Goal: Extract Z_IB = X ∩ Y (intersection)
    """
    
    def __init__(self, compression_ratio: float = 0.5, beta: float = 1.0):
        """
        Args:
            compression_ratio: Target compression ratio (0-1)
            beta: Compression weight (higher = more compression)
        """
        self.compression_ratio = compression_ratio
        self.beta = beta
        
        # Initialize Gemini for semantic filtering
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

        # Together AI fallback
        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None

        logger.info(f"Information Bottleneck Service initialized (ratio={compression_ratio}, beta={beta})")
    
    def _estimate_relevance_scores(self, sentences: List[str], query: str) -> List[float]:
        """
        Estimate relevance of each sentence to the query using LLM.
        
        This approximates I(sentence; answer).
        
        Args:
            sentences: List of sentences from retrieved context
            query: User query (proxy for answer)
            
        Returns:
            List of relevance scores
        """
        if not self.client and not self.together_client:
            # Fallback: simple keyword matching
            return [sum(1 for word in query.lower().split() if word in sent.lower())
                    for sent in sentences]

        sentences_text = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(sentences)])
        prompt = f"""Score each sentence's relevance to answering this query.

Query: {query}

Sentences:
{sentences_text}

Return ONLY a JSON array of scores (0-1 for each sentence):
[score1, score2, ...]
"""

        # Try Gemini first
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                scores = eval(response.text)
                return scores if len(scores) == len(sentences) else [0.5] * len(sentences)
            except Exception as e:
                logger.warning(f"Relevance scoring failed: {e}")

        # Together AI fallback
        if self.together_client:
            try:
                response = self.together_client.chat.completions.create(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\[[\d.,\s]+\]', content)
                if match:
                    scores = json.loads(match.group())
                    return scores if len(scores) == len(sentences) else [0.5] * len(sentences)
            except Exception as e2:
                logger.warning(f"Together AI relevance scoring also failed: {e2}")

        return [0.5] * len(sentences)
    
    def _compute_redundancy_scores(self, sentences: List[str]) -> List[float]:
        """
        Estimate redundancy (self-information) of each sentence.
        
        This approximates I(sentence; X) - how much "noise" it adds.
        
        High redundancy = common/repeated information
        Low redundancy = unique information
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of redundancy scores
        """
        redundancy_scores = []
        
        for i, sent in enumerate(sentences):
            # Count how similar this sentence is to others
            similarity_count = 0
            sent_words = set(sent.lower().split())
            
            for j, other_sent in enumerate(sentences):
                if i != j:
                    other_words = set(other_sent.lower().split())
                    overlap = len(sent_words & other_words) / (len(sent_words) + 1)
                    similarity_count += overlap
            
            # Normalize by number of comparisons
            redundancy = similarity_count / max(len(sentences) - 1, 1)
            redundancy_scores.append(redundancy)
        
        return redundancy_scores
    
    def filter_context(self, retrieved_context: str, query: str) -> str:
        """
        Apply Information Bottleneck filtering to retrieved context.
        
        Process:
        1. Split context into sentences
        2. Score relevance to query: I(sent; Y)
        3. Score redundancy: I(sent; X)
        4. Compute IB score: relevance - β·redundancy
        5. Select top sentences
        6. Reconstruct compressed context
        
        Args:
            retrieved_context: Original retrieved text
            query: User query
            
        Returns:
            Compressed, filtered context
        """
        logger.info("Applying Information Bottleneck filtering")
        
        # Split into sentences (simple split - could use spaCy for better results)
        sentences = [s.strip() + "." for s in retrieved_context.split(".") if s.strip()]
        
        if not sentences:
            return retrieved_context
        
        # Compute relevance scores: I(Z; Y)
        relevance_scores = self._estimate_relevance_scores(sentences, query)
        
        # Compute redundancy scores: I(Z; X)
        redundancy_scores = self._compute_redundancy_scores(sentences)
        
        # Compute Information Bottleneck scores
        ib_scores = [
            rel - self.beta * red
            for rel, red in zip(relevance_scores, redundancy_scores)
        ]
        
        # Determine how many sentences to keep
        target_count = max(1, int(len(sentences) * self.compression_ratio))
        
        # Select top sentences by IB score
        scored_sentences = list(zip(sentences, ib_scores, range(len(sentences))))
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Keep top sentences and maintain order
        selected = sorted_sentences[:target_count]
        selected = sorted(selected, key=lambda x: x[2])  # Restore original order
        
        # Reconstruct context
        filtered_context = " ".join([sent for sent, _, _ in selected])
        
        original_length = len(retrieved_context)
        filtered_length = len(filtered_context)
        compression_achieved = 1 - (filtered_length / original_length)
        
        logger.info(f"Compressed from {original_length} to {filtered_length} chars "
                   f"({compression_achieved:.1%} reduction)")
        
        return filtered_context
    
    def filter_multiple_contexts(self, contexts: List[str], query: str, 
                                 max_total_tokens: int = 2000) -> str:
        """
        Filter and merge multiple retrieved contexts.
        
        This is useful when you have multiple document chunks
        and need to compress them into a coherent context window.
        
        Args:
            contexts: List of retrieved context strings
            query: User query
            max_total_tokens: Maximum tokens in final output
            
        Returns:
            Merged and filtered context
        """
        if not contexts:
            return ""
        
        # Filter each context individually
        filtered_contexts = [self.filter_context(ctx, query) for ctx in contexts]
        
        # Merge
        merged = "\n\n".join(filtered_contexts)
        
        # If still too long, apply second-level filtering
        approx_tokens = len(merged.split())
        if approx_tokens > max_total_tokens:
            logger.info(f"Applying second-level compression ({approx_tokens} > {max_total_tokens} tokens)")
            
            # Adjust compression ratio to meet token limit
            self.compression_ratio = max_total_tokens / approx_tokens
            merged = self.filter_context(merged, query)
        
        return merged
    
    def explain_filtering(self, retrieved_context: str, query: str) -> Dict:
        """
        Provide detailed explanation of filtering decisions.
        
        Useful for debugging and understanding what information was kept/removed.
        
        Args:
            retrieved_context: Original context
            query: User query
            
        Returns:
            Dictionary with filtering analysis
        """
        sentences = [s.strip() + "." for s in retrieved_context.split(".") if s.strip()]
        
        relevance_scores = self._estimate_relevance_scores(sentences, query)
        redundancy_scores = self._compute_redundancy_scores(sentences)
        ib_scores = [rel - self.beta * red for rel, red in zip(relevance_scores, redundancy_scores)]
        
        # Sort by IB score
        scored = sorted(zip(sentences, ib_scores, relevance_scores, redundancy_scores),
                       key=lambda x: x[1], reverse=True)
        
        target_count = max(1, int(len(sentences) * self.compression_ratio))
        
        kept = scored[:target_count]
        removed = scored[target_count:]
        
        return {
            "total_sentences": len(sentences),
            "kept_count": len(kept),
            "removed_count": len(removed),
            "compression_ratio_achieved": len(removed) / len(sentences),
            "kept_sentences": [
                {
                    "text": sent[:100] + "..." if len(sent) > 100 else sent,
                    "ib_score": score,
                    "relevance": rel,
                    "redundancy": red
                }
                for sent, score, rel, red in kept
            ],
            "removed_sentences": [
                {
                    "text": sent[:100] + "..." if len(sent) > 100 else sent,
                    "ib_score": score,
                    "reason": "low_relevance" if rel < 0.3 else "high_redundancy"
                }
                for sent, score, rel, red in removed[:5]  # Show top 5 removed
            ]
        }
    
    def adaptive_filtering(self, retrieved_context: str, query: str, 
                          quality_threshold: float = 0.7) -> str:
        """
        Apply adaptive compression based on context quality.
        
        If retrieved context is already high-quality (low noise),
        compress less. If noisy, compress more aggressively.
        
        Args:
            retrieved_context: Original context
            query: User query
            quality_threshold: Quality score above which to reduce compression
            
        Returns:
            Adaptively filtered context
        """
        sentences = [s.strip() + "." for s in retrieved_context.split(".") if s.strip()]
        
        if not sentences:
            return retrieved_context
        
        # Estimate overall quality
        relevance_scores = self._estimate_relevance_scores(sentences, query)
        avg_relevance = np.mean(relevance_scores)
        
        logger.info(f"Context quality score: {avg_relevance:.2f}")
        
        # Adjust compression ratio based on quality
        if avg_relevance >= quality_threshold:
            # High quality - compress less
            adjusted_ratio = min(1.0, self.compression_ratio * 1.5)
            logger.info(f"High quality detected, reducing compression to {adjusted_ratio:.2f}")
        else:
            # Low quality - compress more
            adjusted_ratio = max(0.2, self.compression_ratio * 0.7)
            logger.info(f"Low quality detected, increasing compression to {adjusted_ratio:.2f}")
        
        # Temporarily adjust ratio
        original_ratio = self.compression_ratio
        self.compression_ratio = adjusted_ratio
        
        filtered = self.filter_context(retrieved_context, query)
        
        # Restore original ratio
        self.compression_ratio = original_ratio
        
        return filtered
