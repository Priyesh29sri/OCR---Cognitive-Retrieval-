"""
Multi-Armed Bandit Retrieval Service for ICDI-X
===============================================
Dynamically selects between different retrieval methods using reinforcement learning.

Treats retrievers as "arms" and learns which works best for different queries.
Research basis: MBA-RAG
"""

import os
import json
import numpy as np
from typing import List, Dict, Callable
from pathlib import Path
from loguru import logger


MAB_STATE_PATH = Path(os.getenv("MAB_STATE_PATH", "/tmp/mab_state.json"))


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
        self.arms = {
            "dense":        {"alpha": 1.0, "beta": 1.0, "name": "Dense Vector"},
            "sparse":       {"alpha": 1.0, "beta": 1.0, "name": "Sparse/BM25"},
            "graph":        {"alpha": 1.0, "beta": 1.0, "name": "Graph Reasoning"},
            "hierarchical": {"alpha": 1.0, "beta": 1.0, "name": "Hierarchical"}
        }
        self.history = []
        self._load_state()
        logger.info("Multi-Armed Bandit Retrieval initialized")

    def _load_state(self):
        """Load persisted arm stats from disk (cross-session learning)."""
        if MAB_STATE_PATH.exists():
            try:
                with open(MAB_STATE_PATH) as f:
                    saved = json.load(f)
                for arm, params in saved.get("arms", {}).items():
                    if arm in self.arms:
                        self.arms[arm]["alpha"] = params.get("alpha", 1.0)
                        self.arms[arm]["beta"]  = params.get("beta",  1.0)
                self.history = saved.get("history", [])
                logger.info(f"MAB state loaded from {MAB_STATE_PATH} ({len(self.history)} prior trials)")
            except Exception as e:
                logger.warning(f"Could not load MAB state: {e}")

    def _save_state(self):
        """Persist arm stats to disk after every update."""
        try:
            MAB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "arms":    {k: {"alpha": v["alpha"], "beta": v["beta"]} for k, v in self.arms.items()},
                "history": self.history[-500:],  # keep last 500 trials
            }
            with open(MAB_STATE_PATH, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save MAB state: {e}")
    
    def select_arm(self) -> str:
        """Select retrieval arm using Thompson Sampling"""
        samples = {}
        for arm_name, params in self.arms.items():
            sample = np.random.beta(params["alpha"], params["beta"])
            samples[arm_name] = sample
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
        self.arms[arm_name]["alpha"] += reward
        self.arms[arm_name]["beta"]  += (1 - reward)
        self.history.append({"arm": arm_name, "reward": reward})
        logger.info(f"Updated {arm_name}: reward={reward:.3f}")
        self._save_state()
    
    def get_arm_statistics(self) -> Dict:
        """Get statistics for all arms"""
        stats = {}
        for arm_name, params in self.arms.items():
            mean = params["alpha"] / (params["alpha"] + params["beta"])
            stats[arm_name] = {
                "name": params["name"],
                "expected_reward": mean,
                "alpha": params["alpha"],
                "beta":  params["beta"],
                "total_trials": params["alpha"] + params["beta"] - 2
            }
        return stats

    def get_convergence_history(self) -> List[Dict]:
        """Return per-trial history for convergence plots in paper."""
        return self.history
