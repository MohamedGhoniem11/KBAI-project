# Culinary RAG Assistant

A RAG-based assistant that answers culinary questions using your PDF knowledge base.

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   Create a `.env` file in the project folder with your Groq API key:

   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=your-groq-api-key-here
   LLM_MODEL=llama3-8b-8192
   ```

3. **Build the knowledge base:**

   ```bash
   python rebuild_and_test.py
   ```

   (This processes your PDFs and creates the vector store)

4. **Launch the app:**

   ```bash
   streamlit run app.py
   ```

5. **Ask questions** like "How do I make fresh pasta?" in the chat interface.
