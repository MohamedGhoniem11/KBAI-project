# Standalone ingestion — no LangChain document loaders, uses pypdf and python-docx directly.
from pathlib import Path
from typing import List, Dict


def get_document_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    pdf_files = list(directory.glob("*.pdf"))
    docx_files = list(directory.glob("*.docx"))
    all_files = pdf_files + docx_files
    if not all_files:
        raise ValueError(f"No PDF/DOCX files found in {directory}")
    return all_files


def load_pdf(path: Path) -> List[Dict]:
    """Load PDF using pypdf directly (no PyPDFLoader wrapper)."""
    from pypdf import PdfReader  # Direct PDF parsing library
    documents = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 20:
                documents.append({
                    "content": text.strip(),
                    "source": path.name,
                    "page": i + 1
                })
    except Exception as e:
        print(f"  Error loading {path.name}: {e}")
    return documents


def load_docx(path: Path) -> List[Dict]:
    """Load DOCX using python-docx directly (no Docx2txtLoader wrapper)."""
    from docx import Document as DocxDocument  # Direct DOCX parsing library
    documents = []
    try:
        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        if text and len(text.strip()) > 20:
            documents.append({
                "content": text.strip(),
                "source": path.name,
                "page": 1  # DOCX doesn't have pages; treat entire doc as one "page"
            })
    except Exception as e:
        print(f"  Error loading {path.name}: {e}")
    return documents


def load_documents(file_paths: List[Path]) -> List[Dict]:
    documents = []
    for path in file_paths:
        if path.suffix.lower() == ".pdf":
            docs = load_pdf(path)
        elif path.suffix.lower() == ".docx":
            docs = load_docx(path)
        else:
            continue
        documents.extend(docs)
        print(f"  Loaded {path.name}: {len(docs)} pages")
    return documents


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Simple sliding-window text splitter (no RecursiveCharacterTextSplitter dependency)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(text):
            break
        start += chunk_size - overlap  # Slide window forward by (chunk_size - overlap)
    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = 512, overlap: int = 64) -> List[Dict]:
    """Split all documents into chunks using the simple sliding-window splitter."""
    chunks = []
    chunk_id = 0
    for doc in documents:
        text_chunks = split_text(doc["content"], chunk_size, overlap)
        for tc in text_chunks:
            chunks.append({
                "content": tc,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": chunk_id
            })
            chunk_id += 1
    return chunks


def ingest_documents() -> List[Dict]:
    """Full standalone ingestion: find files → load → manual chunk → return list of dicts."""
    from .config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    file_paths = get_document_files(DOCS_DIR)
    print(f"Found {len(file_paths)} PDF/DOCX files")
    documents = load_documents(file_paths)
    print(f"Total pages/sections loaded: {len(documents)}")
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Split into {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    chunks = ingest_documents()
    if chunks:
        print(f"\nSample chunk 0:\n{chunks[0]['content'][:200]}...")
        print(f"Source: {chunks[0]['source']}, Page: {chunks[0]['page']}")
