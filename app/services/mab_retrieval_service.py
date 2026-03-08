"""
Multi-Armed Bandit Retrieval Service for ICDI-X
===============================================
Dynamically selects between different retrieval methods using reinforcement learning.

Treats retrievers as "arms" and learns which works best for different queries.
Research basis: MBA-RAG
"""

import numpy as np
from typing import List, Dict, Callable
from loguru import logger


class MultiArmedBanditRetrieval:
    """
    Thompson Sampling-based retrieval arm selection.
    
    Arms:
    - Dense vector retrieval
    - Sparse (BM25) retrieval
    - Graph retrieval
    - Hierarchical retrieval
    """
    
    def __init__(self):
        # Initialize beta distributions for each arm (alpha, beta parameters)
        self.arms = {
            "dense": {"alpha": 1.0, "beta": 1.0, "name": "Dense Vector"},
            "sparse": {"alpha": 1.0, "beta": 1.0, "name": "Sparse/BM25"},
            "graph": {"alpha": 1.0, "beta": 1.0, "name": "Graph Reasoning"},
            "hierarchical": {"alpha": 1.0, "beta": 1.0, "name": "Hierarchical"}
        }
        
        self.history = []
        logger.info("Multi-Armed Bandit Retrieval initialized")
    
    def select_arm(self) -> str:
        """Select retrieval arm using Thompson Sampling"""
        samples = {}
        
        for arm_name, params in self.arms.items():
            # Sample from beta distribution
            sample = np.random.beta(params["alpha"], params["beta"])
            samples[arm_name] = sample
        
        # Select arm with highest sample
        selected = max(samples.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected arm: {self.arms[selected]['name']}")
        
        return selected
    
    def update_arm(self, arm_name: str, reward: float):
        """
        Update arm statistics based on reward.
        
        Args:
            arm_name: Arm that was used
            reward: Reward value (0-1, higher is better)
        """
        if arm_name not in self.arms:
            return
        
        # Update beta distribution parameters
        self.arms[arm_name]["alpha"] += reward
        self.arms[arm_name]["beta"] += (1 - reward)
        
        self.history.append({"arm": arm_name, "reward": reward})
        logger.info(f"Updated {arm_name}: reward={reward:.3f}")
    
    def get_arm_statistics(self) -> Dict:
        """Get statistics for all arms"""
        stats = {}
        
        for arm_name, params in self.arms.items():
            # Mean of beta distribution = alpha / (alpha + beta)
            mean = params["alpha"] / (params["alpha"] + params["beta"])
            
            stats[arm_name] = {
                "name": params["name"],
                "expected_reward": mean,
                "alpha": params["alpha"],
                "beta": params["beta"],
                "total_trials": params["alpha"] + params["beta"] - 2
            }
        
        return stats
