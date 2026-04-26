# =============================================================================
# LLM Integration Module (FR-04, FR-05)
# =============================================================================
# Generates structured, grounded responses with inline citations (no disclaimers)

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .config import DOMAIN_DISCLAIMER
from .retrieval import RetrievedChunk


class LLMGenerator:
    """LLM integration for grounded generation (FR-04)."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self.llm = self._init_llm()
        self.prompt = self._build_prompt_template()
    
    def _init_llm(self):
        """Initialize LLM."""
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not set - using fallback")
                return None
            return ChatOpenAI(model=self.model, api_key=api_key, temperature=0.1)
        elif self.provider == "groq":
            try:
                from langchain_groq import ChatGroq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    print("Warning: GROQ_API_KEY not set - using fallback mode")
                    return None
                return ChatGroq(
                    model=self.model,
                    api_key=api_key,
                    temperature=0.1
                )
            except ImportError:
                print("Error: langchain-groq not installed. Run: pip install langchain-groq")
                return None
        return None
    
    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Build structured prompt template for consistent answers."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a culinary expert assistant. Your response must be based ONLY on the provided retrieved context.
            
RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the user's question clearly and concisely using ONLY the context above
2. Do not hallucinate or add information not present in the context
3. Cite sources inline using [Source X] format matching the context labels
4. Structure your response with:
   - Direct answer to the question first
   - Additional details from context
   - List of relevant tips if applicable
5. If no relevant context is found, state that clearly
6. Do not include any disclaimers (these are shown separately in the UI)"""),
            ("user", "USER QUESTION: {query}")
        ])
    
    def generate(self, query: str, chunks: List[RetrievedChunk], citations: List[Dict]) -> str:
        """Generate structured response with citations (FR-04, FR-05)."""
        
        if not self.llm:
            return self._generate_fallback(query, chunks, citations)
        
        # Build context from retrieved chunks with source labels
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}] (Page {chunk.page}, {chunk.source}): {chunk.content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        
        try:
            # Generate response using prompt template
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "query": query})
            content = response.content.strip()
            
            # Add formatted sources section
            citations_text = self._format_citations(citations)
            result = f"{content}\n\n---\n**Sources:**\n{citations_text}"
            
            return result
            
        except Exception as e:
            print(f"LLM error: {e}")
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
    
    def _generate_fallback(self, query: str, chunks: List[RetrievedChunk], citations: List[Dict]) -> str:
        """Improved fallback without LLM."""
        if not chunks:
            return "I don't have relevant information about that in my culinary knowledge base. Try asking about recipes, cooking techniques, or food safety."
        
        lines = ["Here's what I found in my knowledge base:\n"]
        
        for i, chunk in enumerate(chunks[:5]):
            content = chunk.content.strip()[:300]
            lines.append(f"**[Source {i+1}]** (Page {chunk.page}, {chunk.source}):\n{content}...\n")
        
        return "\n".join(lines)


def create_llm_generator(provider: str = "openai", model: str = "gpt-4o-mini") -> LLMGenerator:
    """Factory function."""
    return LLMGenerator(provider, model)
