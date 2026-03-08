# app/services/agent_service.py
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

# 1. Define the State (The clipboard the agents share)
class AgentState(TypedDict):
    query: str
    vision_context: str  # Mocked image data for now
    text_context: str    # Mocked RAG data for now
    vision_analysis: str
    text_analysis: str
    final_answer: str
    confidence_score: int

class AgentService:
    def __init__(self):
        # Using free Gemini 1.5 Flash for the agents
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found - agent service will not work")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            api_key=api_key
        )
        self.graph = self._build_graph()
        logger.info("Agent Service initialized with LangGraph multi-agent workflow")

    def _build_graph(self):
        """Constructs the LangGraph Multi-Agent workflow."""
        workflow = StateGraph(AgentState)

        # Add the 3 specialized agents as nodes
        workflow.add_node("vision_agent", self.vision_node)
        workflow.add_node("text_agent", self.text_node)
        workflow.add_node("fusion_agent", self.fusion_node)

        # Define the routing: Start -> Vision & Text (Parallel) -> Fusion -> End
        workflow.set_entry_point("vision_agent")
        workflow.add_edge("vision_agent", "text_agent") 
        workflow.add_edge("text_agent", "fusion_agent")
        workflow.add_edge("fusion_agent", END)

        return workflow.compile()

    # --- AGENT DEFINITIONS ---
    def vision_node(self, state: AgentState):
        logger.info("Vision Agent is analyzing visual context...")
        prompt = f"You are a Vision Expert. Analyze this visual context regarding the query: '{state['query']}'. Context: {state['vision_context']}"
        response = self.llm.invoke(prompt)
        return {"vision_analysis": response.content}

    def text_node(self, state: AgentState):
        logger.info("Text Agent is analyzing text context...")
        prompt = f"You are a Language Expert. Analyze this text document regarding the query: '{state['query']}'. Context: {state['text_context']}"
        response = self.llm.invoke(prompt)
        return {"text_analysis": response.content}

    def fusion_node(self, state: AgentState):
        logger.info("Fusion Agent is synthesizing the final response...")
        prompt = f"""
        You are the Lead Validation Manager. Combine the Vision and Text reports to answer the user's query: '{state['query']}'.
        Vision Report: {state['vision_analysis']}
        Text Report: {state['text_analysis']}
        
        Provide a final, cohesive answer. Then, on a new line, provide a Confidence Score (0-100) based on how well the reports agree.
        Format:
        ANSWER: [Your answer]
        CONFIDENCE: [Score]%
        """
        response = self.llm.invoke(prompt)
        # Parse the output into answer and score
        content = response.content
        parts = content.split("CONFIDENCE:")
        answer = parts[0].replace("ANSWER:", "").strip()
        score = 80  # Default score
        
        if len(parts) > 1:
            try:
                score_str = parts[1].replace("%", "").strip()
                score = int(score_str)
            except:
                pass
        
        return {"final_answer": answer, "confidence_score": score}

    async def process_query_live(self, query: str, websocket):
        """Runs the graph and streams updates to the WebSocket."""
        # Initial mock state (In Phase 5 we will plug your real Qdrant/PageIndex here)
        state = AgentState(
            query=query,
            vision_context="A chart showing a 40% increase in APT attacks detected by Flash-IDS++.",
            text_context="The document states that Flash-IDS++ was evaluated on the DARPA dataset.",
            vision_analysis="", 
            text_analysis="", 
            final_answer="", 
            confidence_score=0
        )

        # Stream the agent thoughts live to the user!
        await websocket.send_text("🧠 **System:** Agents initiated. Routing query...")
        
        for output in self.graph.stream(state):
            if "vision_agent" in output:
                await websocket.send_text(f"👁️ **Vision Agent:** {output['vision_agent']['vision_analysis'][:200]}...")
            elif "text_agent" in output:
                await websocket.send_text(f"📝 **Text Agent:** {output['text_agent']['text_analysis'][:200]}...")
            elif "fusion_agent" in output:
                result = output['fusion_agent']
                await websocket.send_text(f"\n🎯 **Fusion Manager (Final Answer):**\n{result['final_answer']}")
                await websocket.send_text(f"📊 **System Confidence Score:** {result['confidence_score']}%")
