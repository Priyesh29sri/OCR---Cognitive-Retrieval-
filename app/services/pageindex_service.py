import os
import json
from google import genai
from together import Together
from loguru import logger

class PageIndexService:
    def __init__(self):
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            logger.warning("GEMINI_API_KEY not found. PageIndex will not run.")
            self.client = None

        # Together AI fallback
        together_key = os.getenv("TOGETHER_API_KEY")
        self.together_client = Together(api_key=together_key) if together_key else None

    async def build_document_tree(self, document_text: str, filename: str) -> dict:
        """
        Mimics the PageIndex methodology: Uses an LLM to read the entire document 
        and build a hierarchical semantic tree structure for reasoning-based retrieval.
        """
        if not self.client or not document_text.strip():
            return {}

        logger.info(f"Building PageIndex tree for {filename} using Gemini...")

        # We force the LLM to output a strict JSON tree structure
        prompt = f"""
        You are an expert document analyzer. Analyze the following document text and build a 
        semantic hierarchical 'Table of Contents' tree structure.
        
        Return ONLY a valid JSON object representing the root of the document.
        Use this exact schema for nodes:
        {{
            "title": "Document Title",
            "summary": "Overall document summary",
            "nodes": [
                {{
                    "title": "Section Title",
                    "summary": "1-sentence summary of this specific section",
                    "nodes": [] 
                }}
            ]
        }}
        
        Document Text:
        {document_text}
        """

        try:
            response_text = None

            # Try Gemini first
            if self.client:
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config={"response_mime_type": "application/json"}
                    )
                    response_text = response.text.strip()
                    logger.info(f"PageIndex: Gemini response length: {len(response_text)} chars")
                except Exception as gemini_err:
                    logger.warning(f"PageIndex: Gemini failed ({gemini_err}), falling back to Together AI")

            # Fallback to Together AI (Llama 3.3 70B)
            if not response_text and self.together_client:
                try:
                    resp = self.together_client.chat.completions.create(
                        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1500,
                    )
                    response_text = resp.choices[0].message.content.strip()
                    logger.info(f"PageIndex: Together AI response length: {len(response_text)} chars")
                except Exception as together_err:
                    logger.error(f"PageIndex: Together AI also failed: {together_err}")

            if not response_text:
                return {}
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            tree_data = json.loads(response_text)
            
            # Save the tree locally to act as our "Vectorless Database"
            tree_filename = f"{filename}_pageindex.json"
            with open(tree_filename, "w") as f:
                json.dump(tree_data, f, indent=4)
                
            logger.info(f"PageIndex tree successfully built and saved as {tree_filename}")
            return tree_data
            
        except Exception as e:
            logger.error(f"Failed to build PageIndex tree: {str(e)}")
            return {}