from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from dotenv import load_dotenv
load_dotenv()
# Load model once at module level to avoid reloading on every call
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model (singleton pattern).
    Avoids reloading the model on every function call.
    """
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embeddings(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of text chunks to embed.
        batch_size: Number of texts to process at once (tune for memory).

    Returns:
        numpy array of shape (len(texts), embedding_dim)
        embedding_dim = 384 for all-MiniLM-L6-v2
    """
    model = get_model()
    print(f"[Embeddings] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine similarity = dot product
    )
    print(f"[Embeddings] Done. Shape: {embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for retrieval.

    Args:
        query: User's natural language question.

    Returns:
        1-D numpy array of shape (embedding_dim,)
    """
    model = get_model()
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding[0]   # Return 1-D array