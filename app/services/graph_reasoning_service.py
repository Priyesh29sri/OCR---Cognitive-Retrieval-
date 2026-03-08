"""
Graph Reasoning Engine for ICDI-X
==================================
NeuroPath-inspired semantic path reasoning for multi-hop queries.

Inspired by hippocampal place cells, this engine maps entities to "place cells"
and uses goal-directed semantic path tracking for complex reasoning.

Research basis: NeuroPath (2025), HippoRAG
"""

import os
from typing import List, Dict, Optional, Tuple
from loguru import logger
from google import genai
from app.services.knowledge_graph_service import KnowledgeEntity, KnowledgeRelation, KnowledgeGraphService


class SemanticPath:
    """Represents a reasoning path through the knowledge graph"""
    def __init__(self):
        self.relations: List[KnowledgeRelation] = []
        self.entities: List[KnowledgeEntity] = []
        self.confidence_score: float = 1.0
        self.reasoning_chain: List[str] = []
    
    def add_step(self, relation: KnowledgeRelation, reasoning: str):
        """Add a reasoning step to the path"""
        self.relations.append(relation)
        self.entities.append(relation.target)
        self.reasoning_chain.append(reasoning)
        self.confidence_score *= relation.confidence
    
    def get_path_summary(self) -> str:
        """Get human-readable path summary"""
        if not self.relations:
            return "Empty path"
        
        path_str = self.relations[0].source.name
        for relation in self.relations:
            path_str += f" --[{relation.relation_type}]--> {relation.target.name}"
        
        return path_str
    
    def __repr__(self):
        return f"SemanticPath({self.get_path_summary()}, confidence={self.confidence_score:.2f})"


class GraphReasoningService:
    """
    NeuroPath-inspired reasoning engine for multi-hop queries.
    
    Uses semantic path tracking to navigate knowledge graphs and answer
    complex queries that require connecting multiple pieces of information.
    
    Key Features:
    - Preplay: Goal-directed path expansion from seed entities
    - Replay: Post-retrieval completion of reasoning chains
    - Dynamic pruning of irrelevant paths
    - LLM-guided path filtering
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraphService):
        self.kg = knowledge_graph
        
        # Initialize Gemini for path evaluation
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None
        
        logger.info("Graph Reasoning Engine initialized")
    
    def identify_seed_entities(self, query: str) -> List[KnowledgeEntity]:
        """
        Extract seed entities from the query.
        
        Args:
            query: User query
            
        Returns:
            List of seed entities found in knowledge graph
        """
        seeds = []
        query_lower = query.lower()
        
        # Simple keyword matching - in production, use NER
        for entity in self.kg.entities.values():
            if entity.name.lower() in query_lower:
                seeds.append(entity)
        
        logger.info(f"Identified {len(seeds)} seed entities: {[e.name for e in seeds]}")
        return seeds
    
    def evaluate_path_relevance(self, path: SemanticPath, query: str) -> float:
        """
        Use LLM to evaluate if a reasoning path is relevant to the query.
        
        Args:
            path: Candidate reasoning path
            query: Original query
            
        Returns:
            Relevance score (0-1)
        """
        if not self.client:
            return 0.5  # Default score if no LLM available
        
        try:
            prompt = f"""Evaluate if this reasoning path helps answer the query.

Query: {query}

Reasoning Path:
{path.get_path_summary()}

Reasoning Chain:
{' -> '.join(path.reasoning_chain)}

Is this path relevant? Respond with ONLY a number between 0 and 1.
0 = completely irrelevant
1 = directly answers the query
"""
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            score = float(response.text.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Path evaluation failed: {e}")
            return 0.5
    
    def expand_path_preplay(
        self, 
        current_entity: KnowledgeEntity,
        query: str,
        current_path: SemanticPath,
        max_depth: int = 3,
        visited: Optional[set] = None
    ) -> List[SemanticPath]:
        """
        Preplay phase: Expand paths from current entity using LLM guidance.
        
        This is the core NeuroPath algorithm - using LLM to predict promising
        directions and prune noisy paths.
        
        Args:
            current_entity: Current position in graph
            query: Target query
            current_path: Path taken so far
            max_depth: Maximum reasoning depth
            visited: Set of visited entities
            
        Returns:
            List of completed reasoning paths
        """
        if visited is None:
            visited = set()
        
        entity_key = f"{current_entity.name}_{current_entity.entity_type}"
        
        # Base case: max depth reached
        if len(current_path.relations) >= max_depth:
            return [current_path]
        
        # Base case: already visited this entity
        if entity_key in visited:
            return []
        
        visited.add(entity_key)
        
        # Get outgoing relations
        outgoing_relations = self.kg.adjacency_list.get(entity_key, [])
        
        if not outgoing_relations:
            return [current_path]
        
        # Use LLM to filter promising relations
        promising_relations = self._filter_promising_relations(
            outgoing_relations, 
            query, 
            current_path
        )
        
        # Recursively expand paths
        all_paths = []
        for relation in promising_relations[:3]:  # Limit branching factor
            new_path = SemanticPath()
            new_path.relations = current_path.relations.copy()
            new_path.entities = current_path.entities.copy()
            new_path.reasoning_chain = current_path.reasoning_chain.copy()
            new_path.confidence_score = current_path.confidence_score
            
            # Add reasoning for this step
            reasoning = f"From {relation.source.name} to {relation.target.name} via {relation.relation_type}"
            new_path.add_step(relation, reasoning)
            
            # Recursively expand
            expanded_paths = self.expand_path_preplay(
                relation.target,
                query,
                new_path,
                max_depth,
                visited.copy()
            )
            
            all_paths.extend(expanded_paths)
        
        return all_paths if all_paths else [current_path]
    
    def _filter_promising_relations(
        self,
        relations: List[KnowledgeRelation],
        query: str,
        current_path: SemanticPath
    ) -> List[KnowledgeRelation]:
        """
        Use LLM to filter relations that are likely to lead to the answer.
        
        This implements NeuroPath's "semantic direction prediction".
        """
        if not self.client or len(relations) <= 3:
            return relations
        
        try:
            # Build relation descriptions
            relation_desc = "\n".join([
                f"{i+1}. {r.source.name} --[{r.relation_type}]--> {r.target.name}"
                for i, r in enumerate(relations)
            ])
            
            prompt = f"""Given this query and current reasoning path, which relations are most promising?

Query: {query}

Current Path: {current_path.get_path_summary()}

Available Relations:
{relation_desc}

Select the top 3 most promising relations by listing their numbers (e.g., "1,3,5").
"""
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=filter_prompt
            )
            
            # Parse selected indices
            selected = [int(x.strip()) - 1 for x in response.text.strip().split(",") if x.strip().isdigit()]
            
            return [relations[i] for i in selected if 0 <= i < len(relations)]
            
        except Exception as e:
            logger.warning(f"Relation filtering failed: {e}")
            return relations[:3]
    
    def reason_multi_hop(self, query: str, max_depth: int = 3) -> List[SemanticPath]:
        """
        Perform multi-hop reasoning over the knowledge graph.
        
        This is the main entry point implementing the full NeuroPath algorithm:
        1. Identify seed entities from query
        2. Preplay: Expand paths using LLM guidance
        3. Rank paths by relevance
        
        Args:
            query: User query requiring multi-hop reasoning
            max_depth: Maximum reasoning chain length
            
        Returns:
            Ranked list of reasoning paths
        """
        logger.info(f"Starting multi-hop reasoning for: {query}")
        
        # Step 1: Identify seed entities
        seed_entities = self.identify_seed_entities(query)
        
        if not seed_entities:
            logger.warning("No seed entities found in knowledge graph")
            return []
        
        # Step 2: Preplay - expand paths from each seed
        all_paths = []
        for seed in seed_entities:
            initial_path = SemanticPath()
            initial_path.entities.append(seed)
            
            paths = self.expand_path_preplay(
                seed,
                query,
                initial_path,
                max_depth
            )
            
            all_paths.extend(paths)
        
        # Step 3: Evaluate and rank paths
        for path in all_paths:
            relevance = self.evaluate_path_relevance(path, query)
            path.confidence_score *= relevance
        
        # Sort by confidence
        ranked_paths = sorted(all_paths, key=lambda p: p.confidence_score, reverse=True)
        
        logger.info(f"Generated {len(ranked_paths)} reasoning paths")
        
        return ranked_paths[:5]  # Return top 5 paths
    
    def extract_context_from_path(self, path: SemanticPath) -> str:
        """
        Extract textual context from a reasoning path.
        
        This retrieves the actual text/provenance associated with entities
        and relations in the path for use in answer generation.
        
        Args:
            path: Reasoning path
            
        Returns:
            Concatenated context text
        """
        context_parts = []
        
        # Add context for each entity in path
        for entity in path.entities:
            if entity.attributes.get("description"):
                context_parts.append(f"{entity.name}: {entity.attributes['description']}")
        
        # Add reasoning chain
        context_parts.append(f"Reasoning: {' -> '.join(path.reasoning_chain)}")
        
        return "\n\n".join(context_parts)
    
    def explain_reasoning(self, path: SemanticPath) -> Dict:
        """
        Generate human-readable explanation of the reasoning path.
        
        Args:
            path: Reasoning path
            
        Returns:
            Dictionary with explanation components
        """
        return {
            "path_summary": path.get_path_summary(),
            "confidence": path.confidence_score,
            "steps": [
                {
                    "from": rel.source.name,
                    "relation": rel.relation_type,
                    "to": rel.target.name,
                    "reasoning": reasoning
                }
                for rel, reasoning in zip(path.relations, path.reasoning_chain)
            ],
            "entities_involved": [e.name for e in path.entities],
            "conclusion": path.reasoning_chain[-1] if path.reasoning_chain else "No conclusion"
        }
