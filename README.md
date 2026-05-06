# Culinary RAG System (KBAI Final Project)
RAG-based assistant for culinary arts using LangChain + LangGraph with Grok (xAI) API.

## ✅ Project Compliance
- **FR-01**: Free-form NL query input via CLI/Streamlit
- **FR-02**: `all-MiniLM-L6-v2` embeddings (justified in `src/config.py`)
- **FR-03**: Top-5 chunks, cosine similarity ≥0.65
- **FR-04**: Grok-only grounded generation (no hallucination)
- **FR-05**: Inline citations with source + page number
- **FR-06**: Add PDF/DOCX to KB without retraining (FAISS limitation: deletion requires rebuild)
- **FR-07**: Domain disclaimer appended to every response
- **LangChain + LangGraph**: Stateful agentic pipeline with reflection/re-retrieval

## 📂 Project Structure
```
├── KB/                  # Source PDF/DOCX (not committed to GitHub)
├── data/vectorstore/    # FAISS index (not committed to GitHub)
├── src/
│   ├── config.py         # Thresholds, embedding model, disclaimer
│   ├── ingestion.py      # PDF/DOCX loading + chunking (512 tokens, 64 overlap)
│   ├── embeddings.py     # HuggingFace embeddings (normalized for cosine)
│   ├── vectorstore.py    # FAISS vector store management
│   ├── retrieval.py      # Semantic retrieval (top-5, ≥0.65 cosine)
│   ├── langgraph_pipeline.py  # Agentic graph (retrieve → reflect → re-retrieve → generate)
│   ├── llm_integration.py     # Grok-only LLM with citations + disclaimer
│   └── __init__.py
├── tests/               # Test files
├── app.py               # Streamlit web UI
├── main.py              # CLI test interface (no Streamlit needed)
├── rebuild_and_test.py  # Rebuild vector store from KB documents
├── requirements.txt     # Dependencies
├── .env                 # API keys (not committed)
└── .gitignore           # Excludes vector store, .env, KB PDFs
```

## 🚀 Setup Instructions (For New Machines)
### 1. Prerequisites
- Python 3.10+
- Grok (xAI) API key from [xAI Console](https://console.x.ai/)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create `.env` file (already gitignored):
```env
LLM_PROVIDER=xai
XAI_API_KEY=your_xai_grok_api_key_here
LLM_MODEL=grok-2-1212
```

### 4. Add Source Documents
Place PDF/DOCX files in the `KB/` directory (at least 10 required per project spec).

### 5. Rebuild Vector Store
Run once to ingest documents and create FAISS index:
```bash
python rebuild_and_test.py
```

## 🧪 Testing Without Streamlit
Run CLI tests with Grok API:
```bash
python main.py
```
This will:
1. Initialize the RAG system
2. Run 3 test culinary queries
3. Output generated answers + top 4 retrieved chunks for verification (prevents hallucination)

## 🌐 Web UI (Optional)
```bash
streamlit run app.py
```

## 📝 Key Design Decisions
1. **Embedding Model**: `all-MiniLM-L6-v2` chosen for fast inference + strong culinary term matching
2. **Vector Store**: FAISS for fast ANN search (note: deletion requires rebuild; Chroma recommended for dynamic KB updates)
3. **LangGraph Pipeline**: Looping agentic flow with reflection to ensure retrieval quality before generation
4. **Hallucination Prevention**: Returns top 4 retrieved chunks alongside LLM answer for user verification

## ⚠️ Notes for GitHub Cloners
- `KB/*.pdf`, `data/vectorstore/`, `.env` are not committed to GitHub
- You must add your own PDF/DOCX to `KB/` and run `rebuild_and_test.py` before use
- Get your own Grok API key and set it in `.env`
