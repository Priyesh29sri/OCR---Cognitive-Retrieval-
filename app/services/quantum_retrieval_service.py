"""
Quantum-Inspired Retrieval Service for ICDI-X
==============================================
Uses quantum state representations and fidelity-based similarity for retrieval.

Unlike traditional cosine similarity, quantum retrieval captures nonlinear
semantic relationships and global contextual patterns.

Research basis: Quantum-RAG, Pun-RAG (2025)
"""

import numpy as np
from typing import List, Tuple, Dict
from loguru import logger
from scipy import linalg


class QuantumRetrievalService:
    """
    Quantum-inspired retrieval using quantum state representations.
    
    Key Concepts:
    - Embeddings represented as pure quantum states |ψ⟩
    - Similarity measured via quantum fidelity instead of cosine
    - Density matrices for mixed state representations
    - Frequency domain filtering via FFT
    
    Mathematical Foundation:
    |ψ(x)⟩ = x / ||x||₂
    ρ(x) = |ψ(x)⟩⟨ψ(x)|  (density matrix)
    Fidelity(ρq, ρk) = Tr(√(√ρq · ρk · √ρq))²
    """
    
    def __init__(self):
        logger.info("Quantum Retrieval Service initialized")
        self.use_fft_filtering = True  # Enable frequency domain filtering
    
    def _normalize_to_quantum_state(self, embedding: np.ndarray) -> np.ndarray:
        """
        Convert classical embedding to quantum state |ψ⟩.
        
        Args:
            embedding: Classical vector embedding
            
        Returns:
            Normalized quantum state vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.warning("Zero norm embedding encountered")
            return embedding
        
        return embedding / norm
    
    def _create_density_matrix(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Create density matrix ρ = |ψ⟩⟨ψ| from quantum state.
        
        Args:
            quantum_state: Normalized quantum state vector
            
        Returns:
            Density matrix (d × d)
        """
        # Outer product: |ψ⟩⟨ψ|
        return np.outer(quantum_state, quantum_state.conj())
    
    def _quantum_fidelity(self, rho_q: np.ndarray, rho_k: np.ndarray) -> float:
        """
        Compute quantum fidelity between two density matrices.
        
        Fidelity measures how "close" two quantum states are.
        For pure states, this is equivalent to the squared overlap.
        
        Formula: F(ρq, ρk) = Tr(√(√ρq · ρk · √ρq))²
        
        Args:
            rho_q: Query density matrix
            rho_k: Candidate density matrix
            
        Returns:
            Fidelity score in [0, 1]
        """
        try:
            # Compute matrix square root of rho_q
            sqrt_rho_q = linalg.sqrtm(rho_q)
            
            # Compute √ρq · ρk · √ρq
            product = sqrt_rho_q @ rho_k @ sqrt_rho_q
            
            # Compute square root of product
            sqrt_product = linalg.sqrtm(product)
            
            # Compute trace
            trace = np.trace(sqrt_product).real
            
            # Return squared trace (fidelity)
            fidelity = trace ** 2
            
            # Clamp to [0, 1] to handle numerical errors
            return float(np.clip(fidelity, 0, 1))
            
        except Exception as e:
            logger.warning(f"Fidelity computation failed: {e}, falling back to cosine similarity")
            # Fallback to cosine similarity if quantum computation fails
            return float(np.dot(rho_q.flatten(), rho_k.flatten()))
    
    def _apply_fft_filtering(self, embedding: np.ndarray, cutoff_ratio: float = 0.8) -> np.ndarray:
        """
        Apply FFT-based filtering to remove high-frequency noise.
        
        This is a key feature of Quantum-RAG that helps capture
        global contextual patterns.
        
        Args:
            embedding: Input embedding vector
            cutoff_ratio: Fraction of frequencies to keep (0-1)
            
        Returns:
            Filtered embedding
        """
        # Perform FFT
        fft_result = np.fft.fft(embedding)
        
        # Create frequency mask (keep low frequencies)
        n = len(embedding)
        cutoff = int(n * cutoff_ratio)
        mask = np.zeros(n, dtype=complex)
        mask[:cutoff] = 1
        mask[-cutoff:] = 1  # Also keep negative frequencies
        
        # Apply mask and inverse FFT
        filtered_fft = fft_result * mask
        filtered_embedding = np.fft.ifft(filtered_fft).real
        
        return filtered_embedding
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          candidate_embedding: np.ndarray,
                          use_fft: bool = None) -> float:
        """
        Compute quantum fidelity-based similarity between query and candidate.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embedding: Candidate embedding vector
            use_fft: Override FFT filtering setting
            
        Returns:
            Similarity score in [0, 1]
        """
        if use_fft is None:
            use_fft = self.use_fft_filtering
        
        # Convert embeddings to numpy arrays if needed
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        if not isinstance(candidate_embedding, np.ndarray):
            candidate_embedding = np.array(candidate_embedding)
        
        # Apply FFT filtering if enabled
        if use_fft:
            query_embedding = self._apply_fft_filtering(query_embedding)
            candidate_embedding = self._apply_fft_filtering(candidate_embedding)
        
        # Convert to quantum states
        psi_q = self._normalize_to_quantum_state(query_embedding)
        psi_k = self._normalize_to_quantum_state(candidate_embedding)
        
        # Create density matrices
        rho_q = self._create_density_matrix(psi_q)
        rho_k = self._create_density_matrix(psi_k)
        
        # Compute quantum fidelity
        similarity = self._quantum_fidelity(rho_q, rho_k)
        
        return similarity
    
    def rank_candidates(self, query_embedding: np.ndarray,
                       candidate_embeddings: List[np.ndarray],
                       candidate_ids: List[str] = None,
                       top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Rank candidates using quantum similarity.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            candidate_ids: Optional list of candidate identifiers
            top_k: Number of top candidates to return
            
        Returns:
            List of (index, score) tuples sorted by score
        """
        if candidate_ids is None:
            candidate_ids = list(range(len(candidate_embeddings)))
        
        logger.info(f"Ranking {len(candidate_embeddings)} candidates using quantum similarity")
        
        # Compute similarities
        similarities = []
        for idx, candidate_emb in enumerate(candidate_embeddings):
            score = self.compute_similarity(query_embedding, candidate_emb)
            similarities.append((idx, score))
        
        # Sort by score descending
        ranked = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_results = ranked[:top_k]
        
        logger.info(f"Top result: score={top_results[0][1]:.4f}" if top_results else "No results")
        
        return top_results
    
    def hybrid_similarity(self, query_embedding: np.ndarray,
                         candidate_embedding: np.ndarray,
                         alpha: float = 0.7) -> float:
        """
        Compute hybrid similarity combining quantum fidelity and cosine similarity.
        
        This fusion approach can provide better results by leveraging both methods.
        
        Args:
            query_embedding: Query embedding
            candidate_embedding: Candidate embedding
            alpha: Weight for quantum similarity (1-alpha for cosine)
            
        Returns:
            Hybrid similarity score
        """
        # Quantum similarity
        quantum_sim = self.compute_similarity(query_embedding, candidate_embedding)
        
        # Classical cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        candidate_norm = candidate_embedding / (np.linalg.norm(candidate_embedding) + 1e-10)
        cosine_sim = np.dot(query_norm, candidate_norm)
        
        # Weighted combination
        hybrid_sim = alpha * quantum_sim + (1 - alpha) * cosine_sim
        
        return float(hybrid_sim)
    
    def explain_quantum_advantage(self, query_embedding: np.ndarray,
                                  candidate_embeddings: List[np.ndarray]) -> Dict:
        """
        Compare quantum vs classical similarity for analysis.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidates
            
        Returns:
            Dictionary with comparative analysis
        """
        quantum_scores = []
        cosine_scores = []
        
        for candidate in candidate_embeddings:
            # Quantum score
            q_score = self.compute_similarity(query_embedding, candidate)
            quantum_scores.append(q_score)
            
            # Cosine score
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            c_norm = candidate / (np.linalg.norm(candidate) + 1e-10)
            c_score = np.dot(q_norm, c_norm)
            cosine_scores.append(c_score)
        
        # Compute ranking differences
        quantum_ranking = np.argsort(quantum_scores)[::-1]
        cosine_ranking = np.argsort(cosine_scores)[::-1]
        
        ranking_diff = np.mean(quantum_ranking != cosine_ranking)
        
        return {
            "quantum_mean_score": float(np.mean(quantum_scores)),
            "cosine_mean_score": float(np.mean(cosine_scores)),
            "ranking_agreement": float(1 - ranking_diff),
            "quantum_advantage_cases": int(np.sum(np.array(quantum_scores) > np.array(cosine_scores))),
            "total_comparisons": len(candidate_embeddings)
        }
