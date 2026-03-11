"""
retriever.py
------------
Two responsibilities:
  1. Ingest government-scheme PDFs → chunk → embed → store in FAISS
  2. Retrieve top-5 relevant chunks for a user query
"""
from dotenv import load_dotenv
load_dotenv()
import os
import re
from typing import List, Dict, Any

import numpy as np

# PDF parsing
from pypdf import PdfReader               # pip install pypdf

from embeddings import generate_embeddings, embed_query
from vector_store import build_index, load_index, search

# ── Config ──────────────────────────────────────────────────────────────────
SCHEMES_DIR  = os.getenv("SCHEMES_DIR",  "data/schemes")   # folder with PDFs
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE",   "500"))        # words per chunk
CHUNK_OVERLAP= int(os.getenv("CHUNK_OVERLAP","50"))         # word overlap
TOP_K        = int(os.getenv("TOP_K", "5"))


# ── Document loading ─────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_all_schemes(directory: str = SCHEMES_DIR) -> List[Dict[str, str]]:
    """
    Walk the schemes directory and load text from every PDF.

    Returns:
        List of {"filename": str, "text": str} dicts.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Schemes directory not found: '{directory}'\n"
            "Create it and add government scheme PDFs (see README for sources)."
        )

    docs = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(directory, fname)
            print(f"[Retriever] Loading {fname}...")
            text = load_pdf(fpath)
            docs.append({"filename": fname, "text": text})

    print(f"[Retriever] Loaded {len(docs)} PDF(s) from {directory}")
    return docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:       Full document text.
        chunk_size: Target words per chunk.
        overlap:    Words to repeat between consecutive chunks (context continuity).

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap       # slide window with overlap
    return chunks


# ── Index building ────────────────────────────────────────────────────────────

def build_scheme_index(schemes_dir: str = SCHEMES_DIR) -> None:
    """
    Full pipeline: load PDFs → chunk → embed → store FAISS index.
    Run this once (or whenever the PDFs change).
    """
    docs = load_all_schemes(schemes_dir)
    if not docs:
        raise ValueError(f"No PDF files found in '{schemes_dir}'.")

    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    for doc in docs:
        chunks = split_into_chunks(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "text":     chunk,
                "source":   doc["filename"],
                "chunk_id": i,
            })

    print(f"[Retriever] Total chunks: {len(all_chunks)}")

    # Generate embeddings and build FAISS index
    embeddings = generate_embeddings(all_chunks).astype("float32")
    build_index(embeddings, all_metadata)
    print("[Retriever] Index built and saved successfully.")


# ── Retrieval ─────────────────────────────────────────────────────────────────

# Cache the index in memory across calls (avoid reloading on every query)
_index = None
_metadata = None


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Return the top-k most relevant scheme chunks for a user query.

    Args:
        query: Natural language question / problem description.
        top_k: Number of chunks to retrieve.

    Returns:
        List of dicts: {"text", "source", "chunk_id", "score"}
    """
    global _index, _metadata

    # Lazy-load index on first call
    if _index is None:
        _index, _metadata = load_index()

    query_emb = embed_query(query).astype("float32")
    results = search(_index, _metadata, query_emb, top_k=top_k)
    return results