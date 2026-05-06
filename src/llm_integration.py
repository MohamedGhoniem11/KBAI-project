# =============================================================================
# LLM Integration Module (FR-04, FR-05, FR-07)
# =============================================================================
# Generates grounded, cited responses using Grok (xAI) only
# Appends domain disclaimer to every response

import os
from typing import List, Dict

from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate

from .config import DOMAIN_DISCLAIMER
from .retrieval import RetrievedChunk


class LLMGenerator:
    """Grok-only LLM integration for grounded generation (FR-04)."""
    
    def __init__(self, model: str = "grok-2-1212"):
        self.model = model
        self.llm = self._init_llm()
        self.prompt = self._build_prompt_template()
    
    def _init_llm(self):
        """Initialize Grok (xAI) LLM only."""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key or api_key == "your_xai_grok_api_key_here":
            raise ValueError("XAI_API_KEY not set in .env file")
        
        return ChatXAI(
            model=self.model,
            api_key=api_key,
            temperature=0.1
        )
    
    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Build prompt enforcing grounded generation (FR-04, FR-05)."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a culinary expert assistant. Your response must be based ONLY on the provided retrieved context.
            
RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the user's question clearly and concisely using ONLY the context above
2. Do not hallucinate or add information not present in the context
3. Cite sources inline using [Source X] format matching the context labels
4. If no relevant context is found, state that clearly
5. Do not include any disclaimers (these are appended automatically)"""),
            ("user", "USER QUESTION: {query}")
        ])
    
    def generate(self, query: str, chunks: List[RetrievedChunk], citations: List[Dict]) -> dict:
        """Generate response with citations and disclaimer (FR-04, FR-05, FR-07)."""
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}] (Page {chunk.page}, {chunk.source}): {chunk.content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        try:
            # Generate response using Grok
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "query": query})
            answer = response.content.strip()
            
            # Format citations
            citations_text = self._format_citations(citations)
            
            # Append disclaimer (FR-07)
            full_response = f"{answer}\n\n---\n**Sources:**\n{citations_text}\n\n{DOMAIN_DISCLAIMER}"
            
            return {
                "answer": full_response,
                "retrieved_chunks": [chunk.model_dump() for chunk in chunks[:4]],  # Top 4 chunks for verification
                "citations": citations
            }
            
        except Exception as e:
            print(f"Grok API error: {e}")
            return self._generate_fallback(query, chunks, citations)
    
    def _format_citations(self, citations: List[Dict]) -> str:
        """Format citations with page numbers (FR-05)."""
        if not citations:
            return "No sources found"
        
        lines = []
        for i, cit in enumerate(citations):
            source = cit.get("source", "Unknown")
            page = cit.get("page", "?")
            score = round(cit.get("score", 0), 3)
            lines.append(f"[{i+1}] {source}, Page {page} (Relevance: {score})")
        
        return "\n".join(lines)
    
    def _generate_fallback(self, query: str, chunks: List[RetrievedChunk], citations: List[Dict]) -> dict:
        """Fallback without LLM (returns retrieved chunks for verification)."""
        if not chunks:
            return {
                "answer": "I don't have relevant information in my knowledge base. Try asking about recipes, cooking techniques, or food safety.",
                "retrieved_chunks": [],
                "citations": []
            }
        
        lines = ["Here's what I found in my knowledge base:\n"]
        for i, chunk in enumerate(chunks[:4]):  # Top 4 chunks
            content = chunk.content.strip()[:300]
            lines.append(f"**[Source {i+1}]** (Page {chunk.page}, {chunk.source}):\n{content}...\n")
        
        return {
            "answer": "\n".join(lines),
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks[:4]],
            "citations": citations
        }


def create_llm_generator(model: str = "grok-2-1212") -> LLMGenerator:
    """Factory function for Grok-only LLM generator."""
    return LLMGenerator(model)
