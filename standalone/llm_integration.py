# Standalone LLM integration — uses raw HTTP requests to xAI API (no ChatXAI SDK).
import os
import requests  # Raw HTTP library instead of LangChain's ChatXAI
from typing import List, Dict
from .config import DOMAIN_DISCLAIMER

XAI_API_URL = "https://api.x.ai/v1/chat/completions"  # xAI REST API endpoint


class LLMGenerator:
    """Calls Grok API via raw HTTP POST. No LangChain dependency."""

    def __init__(self, model: str = "grok-3-latest"):
        self.model = model
        self.api_key = os.getenv("XAI_API_KEY")  # Read directly from environment
        if not self.api_key or self.api_key == "your_xai_grok_api_key_here":
            raise ValueError("XAI_API_KEY not set in .env file")

    def generate(self, query: str, chunks: List[Dict], citations: List[Dict]) -> Dict:
        """Build prompt from chunks → POST to xAI API → format response + citations + disclaimer."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source {i+1}] (Page {chunk['page']}, {chunk['source']}): {chunk['content']}"
            )
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        # Manual string formatting instead of ChatPromptTemplate
        system_prompt = (
            "You are a culinary expert assistant. Your response must be based ONLY on the provided retrieved context.\n\n"
            "RETRIEVED CONTEXT:\n"
            f"{context}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question clearly and concisely using ONLY the context above\n"
            "2. Do not hallucinate or add information not present in the context\n"
            "3. Cite sources inline using [Source X] format matching the context labels\n"
            "4. If no relevant context is found, state that clearly\n"
            "5. Do not include any disclaimers (these are appended automatically)"
        )

        try:
            # Raw HTTP request to xAI API (instead of ChatXAI.invoke())
            response = requests.post(
                XAI_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"USER QUESTION: {query}"},
                    ],
                    "temperature": 0.1,  # Low temperature for grounded answers
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()

            citations_text = self._format_citations(citations)
            full_response = f"{answer}\n\n---\n**Sources:**\n{citations_text}\n\n{DOMAIN_DISCLAIMER}"

            return {
                "answer": full_response,
                "retrieved_chunks": chunks[:4],
                "citations": citations,
            }

        except Exception as e:
            print(f"Grok API error: {e}")
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

    def _generate_fallback(self, query: str, chunks: List[Dict], citations: List[Dict]) -> Dict:
        if not chunks:
            return {
                "answer": "I don't have relevant information in my knowledge base. Try asking about recipes, cooking techniques, or food safety.",
                "retrieved_chunks": [],
                "citations": [],
            }
        lines = ["Here's what I found in my knowledge base:\n"]
        for i, chunk in enumerate(chunks[:4]):
            content = chunk["content"].strip()[:300]
            lines.append(
                f"**[Source {i+1}]** (Page {chunk['page']}, {chunk['source']}):\n{content}...\n"
            )
        return {
            "answer": "\n".join(lines),
            "retrieved_chunks": chunks[:4],
            "citations": citations,
        }
