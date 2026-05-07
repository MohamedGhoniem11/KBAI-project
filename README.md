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
├── KB.zip               # Compressed source documents (272MB)
├── KB/                  # Unzipped PDF/DOCX (not committed, gitignored)
├── data/vectorstore/    # FAISS index (not committed, gitignored)
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
├── rebuild_and_test.py  # Rebuild vector store (auto-downloads KB.zip if missing)
├── requirements.txt     # Dependencies
├── .env                 # API key template (committed with placeholder)
└── .gitignore           # Excludes vector store, KB/ folder, IDE files
```

## 🚀 Setup Instructions (For New Machines)
### 1. Prerequisites
- Python 3.10+
- Grok (xAI) API key from [xAI Console](https://console.x.ai/)

### 2. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd assgiment
pip install -r requirements.txt
```

### 3. Configure API Key
The `.env` file is included with a placeholder. Replace the placeholder with your actual Grok API key:
```env
LLM_PROVIDER=xai
XAI_API_KEY=your_actual_grok_api_key_here
LLM_MODEL=grok-2-1212
```

### 4. Set Up Knowledge Base
The source documents are compressed into `KB.zip` (225MB).  
**Download link (Google Drive):**  
[Download KB.zip](https://drive.google.com/uc?export=download&id=1RskXkZXqQiszdQ8QkEYySKlgToQPBZO4)

After downloading, place `KB.zip` in the project root and unzip:
```bash
# Windows (PowerShell)
Expand-Archive KB.zip -DestinationPath KB/

# macOS/Linux
unzip KB.zip -d KB/
```
Alternatively, add your own PDF/DOCX files (at least 10) to the `KB/` folder.

### 5. Rebuild Vector Store
Run once to ingest documents and create FAISS index:
```bash
python rebuild_and_test.py
```
If `KB/` is missing, the script will attempt to download `KB.zip` from the Google Drive link automatically.

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
4. **Hallucination Prevention**: Returns top4 retrieved chunks alongside LLM answer for user verification
5. **KB Compression**: Raw PDFs total 295MB, compressed to 272MB `KB.zip`

## ⚠️ GitHub Notes
- `.env` is committed with a placeholder API key (never commit real keys)
- `KB.zip` is 272MB, exceeding GitHub's 50MB per-file limit. Use Git LFS or external hosting (Google Drive link provided)
- `KB/` folder and `data/vectorstore/` are gitignored (not committed)

## 📎 How to Update the Google Drive Link
1. Upload `KB.zip` to your Google Drive
2. Right-click → Share → Change to "Anyone with the link"
3. Copy the share link (format: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)
4. Extract the `FILE_ID` (the string after `/d/` and before `/view`)
5. Replace `1RskXkZXqQiszdQ8QkEYySKlgToQPBZO4` in this README and in `rebuild_and_test.py` with your actual FILE_ID
