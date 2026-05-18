import os
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from .config import DOMAIN_DISCLAIMER
from .retrieval import RetrievedChunk
from .provider import get_provider, get_api_key, get_default_model, Provider


def _build_prompt_template() -> ChatPromptTemplate:
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


class LLMGenerator:
    def __init__(self, model: Optional[str] = None):
        self.provider = get_provider()
        self.model = model or get_default_model()
        self.api_key = get_api_key()
        self.prompt = _build_prompt_template()
        self.llm = self._init_llm()

    def _init_llm(self):
        if self.provider == Provider.GROQ:
            from langchain_groq import ChatGroq
            return ChatGroq(model=self.model, api_key=self.api_key, temperature=0.1)

        if self.provider == Provider.XAI:
            from langchain_xai import ChatXAI
            return ChatXAI(model=self.model, api_key=self.api_key, temperature=0.1)

        if self.provider == Provider.OLLAMA:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=self.model, temperature=0.1)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, query: str, chunks: List[RetrievedChunk], citations: List[Dict]) -> dict:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}] (Page {chunk.page}, {chunk.source}): {chunk.content}")

        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context, "query": query})
            answer = response.content.strip()

            citations_text = self._format_citations(citations)
            full_response = f"{answer}\n\n---\n**Sources:**\n{citations_text}\n\n{DOMAIN_DISCLAIMER}"

            return {
                "answer": full_response,
                "retrieved_chunks": [chunk.model_dump() for chunk in chunks[:4]],
                "citations": citations
            }

        except Exception as e:
            print(f"LLM API error ({self.provider}): {e}")
            return self._generate_fallback(query, chunks, citations)

    def _format_citations(self, citations: List[Dict]) -> str:
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
        if not chunks:
            return {
                "answer": "I don't have relevant information in my knowledge base. Try asking about recipes, cooking techniques, or food safety.",
                "retrieved_chunks": [],
                "citations": []
            }

        lines = ["Here's what I found in my knowledge base:\n"]
        for i, chunk in enumerate(chunks[:4]):
            content = chunk.content.strip()[:300]
            lines.append(f"**[Source {i+1}]** (Page {chunk.page}, {chunk.source}):\n{content}...\n")

        return {
            "answer": "\n".join(lines),
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks[:4]],
            "citations": citations
        }


def create_llm_generator(model: Optional[str] = None) -> LLMGenerator:
    return LLMGenerator(model)