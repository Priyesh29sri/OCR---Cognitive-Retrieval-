"""
Knowledge Graph Service for ICDI-X
===================================
Extracts entities and relationships from documents to build a knowledge graph
for multi-hop reasoning and relational understanding.

Architecture:
- Entity extraction using LLM
- Relationship identification
- Graph construction and storage
- Graph querying for reasoning

Research inspiration: GraphRAG, NeuroPath
"""

import os
from typing import List, Dict, Tuple, Optional, Set
from loguru import logger
from google import genai
from together import Together


class KnowledgeEntity:
    """Represents an entity in the knowledge graph"""
    def __init__(self, name: str, entity_type: str, attributes: Dict = None):
        self.name = name
        self.entity_type = entity_type
        self.attributes = attributes or {}
        self.provenance = []  # Which documents mention this entity
        
    def __repr__(self):
        return f"Entity(name={self.name}, type={self.entity_type})"
    
    def __hash__(self):
        return hash((self.name, self.entity_type))
    
    def __eq__(self, other):
        if not isinstance(other, KnowledgeEntity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type


class KnowledgeRelation:
    """Represents a relationship between two entities"""
    def __init__(self, source: KnowledgeEntity, relation_type: str, target: KnowledgeEntity, 
                 confidence: float = 1.0):
        self.source = source
        self.relation_type = relation_type
        self.target = target
        self.confidence = confidence
        self.provenance = []
        
    def __repr__(self):
        return f"{self.source.name} --[{self.relation_type}]--> {self.target.name}"


class KnowledgeGraphService:
    """
    Service for constructing and querying knowledge graphs from documents.
    
    Uses LLM-based entity extraction and relationship identification to build
    a structured knowledge representation suitable for multi-hop reasoning.
    """
    
    def __init__(self):
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: List[KnowledgeRelation] = []
        self.adjacency_list: Dict[str, List[KnowledgeRelation]] = {}
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

        # Together AI fallback
        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None
        logger.info("Knowledge Graph Service initialized")
    
    def extract_entities_and_relations(self, text: str, document_id: str) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """
        Extract entities and relationships from text using LLM.
        
        Args:
            text: Document text
            document_id: Source document identifier
            
        Returns:
            Tuple of (entities, relations)
        """
        if not self.client:
            logger.warning("Gemini client not initialized - returning empty results")
            return [], []
        
        try:
            # Construct extraction prompt
            prompt = f"""Extract entities and relationships from the following text.
            
Text:
{text[:3000]}  # Limit to 3000 chars to avoid token limits

Return the results in this exact JSON format:
{{
    "entities": [
        {{"name": "entity_name", "type": "entity_type"}},
        ...
    ],
    "relations": [
        {{"source": "entity1", "relation": "relationship_type", "target": "entity2"}},
        ...
    ]
}}

Entity types: PERSON, ORGANIZATION, TECHNOLOGY, METHOD, DATASET, METRIC, CONCEPT
Relation types: USES, EVALUATED_ON, ACHIEVES, COMPARES_TO, PART_OF, RELATED_TO
"""
            
            response_text = None

            # Try Gemini first
            if self.client:
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    response_text = response.text
                except Exception as gemini_err:
                    logger.warning(f"KnowledgeGraph: Gemini failed ({gemini_err}), falling back to Together AI")

            # Fallback to Together AI
            if not response_text and self.together_client:
                try:
                    resp = self.together_client.chat.completions.create(
                        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        messages=[
                            {"role": "system", "content": "You are a knowledge extraction assistant. Always respond with valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1500,
                    )
                    response_text = resp.choices[0].message.content.strip()
                    # Strip markdown code fences if present
                    if response_text.startswith("```"):
                        response_text = response_text.split("```")[1]
                        if response_text.startswith("json"):
                            response_text = response_text[4:]
                except Exception as together_err:
                    logger.error(f"KnowledgeGraph: Together AI also failed: {together_err}")

            if not response_text:
                return [], []

            result = eval(response_text)  # Parse JSON response
            
            # Create entity objects
            entities = []
            entity_map = {}
            
            for entity_data in result.get("entities", []):
                entity = KnowledgeEntity(
                    name=entity_data["name"],
                    entity_type=entity_data["type"]
                )
                entity.provenance.append(document_id)
                entities.append(entity)
                entity_map[entity.name] = entity
            
            # Create relation objects
            relations = []
            for rel_data in result.get("relations", []):
                source_name = rel_data["source"]
                target_name = rel_data["target"]
                
                if source_name in entity_map and target_name in entity_map:
                    relation = KnowledgeRelation(
                        source=entity_map[source_name],
                        relation_type=rel_data["relation"],
                        target=entity_map[target_name]
                    )
                    relation.provenance.append(document_id)
                    relations.append(relation)
            
            logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations")
            return entities, relations
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []
    
    def add_entities(self, entities: List[KnowledgeEntity]) -> None:
        """Add entities to the knowledge graph"""
        for entity in entities:
            key = f"{entity.name}_{entity.entity_type}"
            if key in self.entities:
                # Merge provenance if entity already exists
                self.entities[key].provenance.extend(entity.provenance)
            else:
                self.entities[key] = entity
        
        logger.info(f"Graph now contains {len(self.entities)} entities")
    
    def add_relations(self, relations: List[KnowledgeRelation]) -> None:
        """Add relations to the knowledge graph and update adjacency list"""
        for relation in relations:
            self.relations.append(relation)
            
            # Update adjacency list for graph traversal
            source_key = f"{relation.source.name}_{relation.source.entity_type}"
            if source_key not in self.adjacency_list:
                self.adjacency_list[source_key] = []
            self.adjacency_list[source_key].append(relation)
        
        logger.info(f"Graph now contains {len(self.relations)} relations")
    
    def build_graph_from_document(self, text: str, document_id: str) -> None:
        """
        Build knowledge graph from a document.
        
        Args:
            text: Full document text
            document_id: Document identifier
        """
        logger.info(f"Building knowledge graph for document: {document_id}")
        
        # Extract entities and relations
        entities, relations = self.extract_entities_and_relations(text, document_id)
        
        # Add to graph
        self.add_entities(entities)
        self.add_relations(relations)
        
        logger.info(f"Knowledge graph updated for {document_id}")
    
    def find_entity(self, entity_name: str) -> Optional[KnowledgeEntity]:
        """Find entity by name"""
        for key, entity in self.entities.items():
            if entity.name.lower() == entity_name.lower():
                return entity
        return None
    
    def get_neighbors(self, entity: KnowledgeEntity, max_hops: int = 1) -> List[KnowledgeEntity]:
        """
        Get neighboring entities within max_hops distance.
        
        Args:
            entity: Source entity
            max_hops: Maximum distance to traverse
            
        Returns:
            List of neighboring entities
        """
        visited: Set[str] = set()
        neighbors: List[KnowledgeEntity] = []
        queue = [(entity, 0)]  # (entity, current_hop)
        
        entity_key = f"{entity.name}_{entity.entity_type}"
        visited.add(entity_key)
        
        while queue:
            current_entity, current_hop = queue.pop(0)
            
            if current_hop >= max_hops:
                continue
            
            current_key = f"{current_entity.name}_{current_entity.entity_type}"
            if current_key in self.adjacency_list:
                for relation in self.adjacency_list[current_key]:
                    target_key = f"{relation.target.name}_{relation.target.entity_type}"
                    
                    if target_key not in visited:
                        visited.add(target_key)
                        neighbors.append(relation.target)
                        queue.append((relation.target, current_hop + 1))
        
        return neighbors
    
    def find_path(self, source_entity: KnowledgeEntity, target_entity: KnowledgeEntity, 
                  max_depth: int = 3) -> Optional[List[KnowledgeRelation]]:
        """
        Find reasoning path between two entities using BFS.
        
        Args:
            source_entity: Starting entity
            target_entity: Target entity
            max_depth: Maximum path length
            
        Returns:
            List of relations forming the path, or None if no path found
        """
        queue = [(source_entity, [])]  # (current_entity, path_so_far)
        visited = set()
        
        source_key = f"{source_entity.name}_{source_entity.entity_type}"
        target_key = f"{target_entity.name}_{target_entity.entity_type}"
        visited.add(source_key)
        
        while queue:
            current_entity, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            current_key = f"{current_entity.name}_{current_entity.entity_type}"
            
            # Check if we reached target
            if current_key == target_key:
                return path
            
            # Explore neighbors
            if current_key in self.adjacency_list:
                for relation in self.adjacency_list[current_key]:
                    neighbor_key = f"{relation.target.name}_{relation.target.entity_type}"
                    
                    if neighbor_key not in visited:
                        visited.add(neighbor_key)
                        new_path = path + [relation]
                        queue.append((relation.target, new_path))
        
        return None
    
    def get_graph_summary(self) -> Dict:
        """Get summary statistics of the knowledge graph"""
        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": list(set(e.entity_type for e in self.entities.values())),
            "relation_types": list(set(r.relation_type for r in self.relations))
        }
    
    def export_graph(self) -> Dict:
        """Export graph in JSON-compatible format"""
        return {
            "entities": [
                {
                    "name": e.name,
                    "type": e.entity_type,
                    "provenance": e.provenance
                }
                for e in self.entities.values()
            ],
            "relations": [
                {
                    "source": r.source.name,
                    "relation": r.relation_type,
                    "target": r.target.name,
                    "confidence": r.confidence
                }
                for r in self.relations
            ]
        }
