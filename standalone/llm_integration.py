import requests

from src.provider import (
    Provider,
    get_api_key,
    get_default_model,
    get_provider,
)

from .config import DOMAIN_DISCLAIMER

OPENAI_COMPATIBLE_URLS = {
    Provider.GROQ: "https://api.groq.com/openai/v1/chat/completions",
    Provider.XAI: "https://api.x.ai/v1/chat/completions",
    Provider.OLLAMA: "http://localhost:11434/v1/chat/completions",
    Provider.OPENAI: "https://api.openai.com/v1/chat/completions",
}


class LLMGenerator:
    """Calls LLM API via raw HTTP. No LangChain dependency."""

    def __init__(self, model: str = None):
        self.provider = get_provider()
        self.model = model or get_default_model()
        self.api_key = get_api_key()
        self.api_url = OPENAI_COMPATIBLE_URLS.get(self.provider)

    def generate(self, query: str, chunks: list[dict], citations: list[dict]) -> dict:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source {i+1}] (Page {chunk['page']}, {chunk['source']}): {chunk['content']}"
            )
        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        system_prompt = (
            "You are a culinary expert assistant. Your response must be based ONLY on "
            "the provided retrieved context.\n\n"
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
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"USER QUESTION: {query}"},
                    ],
                    "temperature": 0.1,
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
            print(f"LLM API error ({self.provider}): {e}")
            return self._generate_fallback(query, chunks, citations)

    def _format_citations(self, citations: list[dict]) -> str:
        if not citations:
            return "No sources found"
        lines = []
        for i, cit in enumerate(citations):
            source = cit.get("source", "Unknown")
            page = cit.get("page", "?")
            score = round(cit.get("score", 0), 3)
            lines.append(f"[{i+1}] {source}, Page {page} (Relevance: {score})")
        return "\n".join(lines)

    def _generate_fallback(self, query: str, chunks: list[dict], citations: list[dict]) -> dict:
        if not chunks:
            no_info_msg = (
                "I don't have relevant information in my knowledge base. "
                "Try asking about recipes, cooking techniques, or food safety."
            )
            return {
                "answer": no_info_msg,
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
