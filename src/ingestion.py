# =============================================================================
# Stage 1: Document Ingestion Pipeline
# =============================================================================
# Loads PDFs, chunks at 512 tokens with 64-token overlap

from pathlib import Path
from typing import List
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Suppress pypdf warnings for cleaner output
warnings.filterwarnings("ignore")


def get_pdf_files(directory: Path) -> List[Path]:
    """Get all PDF files from the documents directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {directory}")
    
    return pdf_files


def load_single_pdf(pdf_path: Path) -> List[Document]:
    """Load a single PDF with robust error handling."""
    documents = []
    print(f"  Loading: {pdf_path.name}")
    
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        for i, doc in enumerate(pages):
            text = doc.page_content.strip()
            if text and len(text) > 20:  # Skip near-empty pages
                doc.page_content = text
                doc.metadata["source"] = pdf_path.name
                doc.metadata["page"] = i + 1  # FR-05: page number
                documents.append(doc)
        
        print(f"    ✅ Loaded {len(documents)} pages from {pdf_path.name}")
        
    except Exception as e:
        print(f"    ⚠️ Error loading {pdf_path.name}: {e}")
        # Try alternative loading method
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip() and len(text.strip()) > 20:
                    doc = Document(
                        page_content=text.strip(),
                        metadata={"source": pdf_path.name, "page": i + 1}
                    )
                    documents.append(doc)
            print(f"    ✅ Loaded {len(documents)} pages (fallback method)")
        except Exception as e2:
            print(f"    ❌ Failed to load {pdf_path.name}: {e2}")
    
    return documents


def load_documents(pdf_files: List[Path]) -> List[Document]:
    """Load all PDF documents with page numbers."""
    documents = []
    
    for pdf_file in pdf_files:
        docs = load_single_pdf(pdf_file)
        documents.extend(docs)
    
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks (512 tokens, 64 overlap)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"Split into {len(chunks)} chunks total")
    return chunks


def ingest_documents() -> List[Document]:
    """Full ingestion pipeline: load PDFs and split into chunks."""
    pdf_files = get_pdf_files(DOCS_DIR)
    print(f"Found {len(pdf_files)} PDF files")
    
    documents = load_documents(pdf_files)
    print(f"Total pages loaded: {len(documents)}")
    
    chunks = split_documents(documents)
    
    print(f"\n✅ Ingestion complete: {len(chunks)} chunks from {len(documents)} pages")
    return chunks


if __name__ == "__main__":
    chunks = ingest_documents()
    if chunks:
        print(f"\nSample chunk 0:\n{chunks[0].page_content[:200]}...")
        print(f"Source: {chunks[0].metadata['source']}")
        print(f"Page: {chunks[0].metadata['page']}")
