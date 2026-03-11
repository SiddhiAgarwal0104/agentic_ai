"""
vector_store.py
---------------
Manages the FAISS vector database:
  - Build index from embeddings
  - Persist index + metadata to disk
  - Load existing index
  - Search for top-k nearest neighbours
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple

# Default paths (can be overridden via environment variables)
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index/schemes.index")
META_PATH  = os.getenv("FAISS_META_PATH",  "data/faiss_index/metadata.pkl")


def build_index(embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (Inner Product) index.
    Because embeddings are L2-normalised, inner product = cosine similarity.

    Args:
        embeddings: Float32 array of shape (n_chunks, embedding_dim).
        metadata:   List of dicts, one per chunk, e.g.:
                    {"text": "...", "source": "pm_kisan.pdf", "chunk_id": 3}

    Returns:
        Populated FAISS index.
    """
    assert embeddings.dtype == np.float32, "FAISS requires float32 embeddings"
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)   # Inner product (= cosine for normalised vecs)
    index.add(embeddings)

    print(f"[VectorStore] Built FAISS index with {index.ntotal} vectors (dim={dim})")

    # Persist index and metadata side-by-side
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[VectorStore] Saved index → {INDEX_PATH}")
    print(f"[VectorStore] Saved metadata → {META_PATH}")

    return index


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """
    Load a previously saved FAISS index and its metadata from disk.

    Returns:
        (index, metadata_list)

    Raises:
        FileNotFoundError: If the index files don't exist yet.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Run build_index() first (or call rag_agent.initialise())."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"[VectorStore] Loaded index with {index.ntotal} vectors")
    return index, metadata


def search(
    index: faiss.IndexFlatIP,
    metadata: List[Dict[str, Any]],
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar chunks for a query embedding.

    Args:
        index:           FAISS index.
        metadata:        Parallel list of chunk metadata.
        query_embedding: 1-D float32 array (embedding_dim,).
        top_k:           Number of results to return.

    Returns:
        List of metadata dicts enriched with a "score" field (cosine similarity).
    """
    # FAISS expects a 2-D array even for a single query
    query_vec = query_embedding.astype(np.float32).reshape(1, -1)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:           # FAISS returns -1 when there aren't enough vectors
            continue
        result = dict(metadata[idx])   # copy to avoid mutating stored metadata
        result["score"] = float(score)
        results.append(result)

    return results