import os
import json
import networkx as nx
import google as genai
from loguru import logger

class GraphService:
    def __init__(self):
        # Configure the free Gemini API for entity extraction
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("GEMINI_API_KEY not found. Graph Service will not run.")
            self.model = None

    async def build_knowledge_graph(self, document_text: str, filename: str) -> dict:
        """
        Extracts entities and relationships from text using an LLM (Step 3) 
        and builds a NetworkX directed graph for the Hybrid Index Layer (Step 4).
        """
        if not self.model or not document_text.strip():
            return {}

        logger.info(f"Extracting Knowledge Graph entities for {filename}...")

        # We force the LLM to output a strict JSON list of triplets
        prompt = f"""
        You are an expert data extractor. Read the following academic text and extract the key entities and their relationships.
        Return ONLY a valid JSON list of dictionaries representing a knowledge graph.
        Use this exact schema for every relationship found:
        [
            {{"subject": "Entity1", "predicate": "relationship_type", "object": "Entity2"}}
        ]
        
        Text:
        {document_text[:15000]} # Limiting context window for extraction speed
        """

        try:
            # 1. LLM Extraction
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            triplets = json.loads(response.text)
            
            # 2. Build the Directed Graph using NetworkX
            G = nx.DiGraph()
            for triplet in triplets:
                # Add nodes and edges (e.g., APT -> detected_by -> Flash-IDS++)
                G.add_edge(triplet["subject"], triplet["object"], relation=triplet["predicate"])
                
            # 3. Save Graph to disk as JSON node-link data (Our local Graph DB)
            graph_data = nx.node_link_data(G)
            graph_filename = f"{filename}_graph.json"
            
            with open(graph_filename, "w") as f:
                json.dump(graph_data, f, indent=4)
                
            logger.info(f"Knowledge Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges successfully built and saved as {graph_filename}")
            return graph_data
            
        except Exception as e:
            logger.error(f"Failed to build Knowledge Graph: {str(e)}")
            return {}