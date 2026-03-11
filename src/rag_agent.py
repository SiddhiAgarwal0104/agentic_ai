import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Any, List

from retriever import retrieve, build_scheme_index
from qa_generator import generate_answer
from eligibility_checker import check_eligibility

# Whether the FAISS index has been initialised this session
_index_ready = False


# ── Initialisation ────────────────────────────────────────────────────────────

def initialise(force_rebuild: bool = False) -> None:
    """
    Prepare the RAG agent:
      - Build FAISS index from PDFs if it doesn't exist (or force_rebuild=True).
      - Train eligibility model if not already saved.

    Call this ONCE at application startup (e.g., from main_graph.py).

    Args:
        force_rebuild: If True, re-index all PDFs even if the index exists.
    """
    global _index_ready

    index_exists = os.path.exists(
        os.getenv("FAISS_INDEX_PATH", "data/faiss_index/schemes.index")
    )

    if force_rebuild or not index_exists:
        print("[RAGAgent] Building scheme index from PDFs...")
        build_scheme_index()
    else:
        print("[RAGAgent] FAISS index already exists. Skipping rebuild.")

    # Trigger eligibility model training / loading
    from eligibility_checker import _get_model
    _get_model()

    _index_ready = True
    print("[RAGAgent] Initialisation complete.")


# ── Main function ─────────────────────────────────────────────────────────────

def recommend_schemes(query: str, user_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve relevant scheme chunks + generate answer
    + check eligibility.

    Args:
        query:     User's natural language question or problem description.
                   Example: "I am a farmer, how can I get financial help?"

        user_info: Dictionary with user profile for eligibility checking.
                   Expected keys:
                     "age"    (int)   – e.g., 45
                     "income" (float) – annual income in INR, e.g., 150000
                     "state"  (str)   – e.g., "bihar"

    Returns:
        {
            "recommended_schemes": List[str],  # eligible scheme category names
            "answer": str,                     # Gemini-generated grounded answer
            "sources": List[str],              # PDF filenames used as context
            "retrieved_chunks": List[dict],    # raw retrieved chunks (for debugging)
        }

    Example:
        result = recommend_schemes(
            query="What schemes are available for elderly poor people?",
            user_info={"age": 68, "income": 75000, "state": "uttar pradesh"}
        )
    """
    global _index_ready

    # Auto-initialise if not done yet (graceful for first call)
    if not _index_ready:
        initialise()

    # ── Step 1: Semantic retrieval ─────────────────────────────────────────
    print(f"\n[RAGAgent] Query: {query}")
    retrieved_chunks: List[Dict[str, Any]] = retrieve(query, top_k=5)
    print(f"[RAGAgent] Retrieved {len(retrieved_chunks)} chunks.")

    # ── Step 2: RAG answer generation via Gemini ───────────────────────────
    qa_result = generate_answer(query, retrieved_chunks)

    # ── Step 3: Eligibility check ──────────────────────────────────────────
    eligible_schemes = check_eligibility(user_info)
    print(f"[RAGAgent] Eligible schemes: {eligible_schemes}")

    # ── Step 4: Assemble final response ───────────────────────────────────
    return {
        "recommended_schemes": eligible_schemes,
        "answer":              qa_result["answer"],
        "sources":             qa_result["sources"],
        "retrieved_chunks":    retrieved_chunks,   # useful for debugging / UI display
    }


# ── CLI test harness ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test — run with: python src/rag_agent.py
    initialise()

    result = recommend_schemes(
        query="I am a 67-year-old retired farmer from Bihar with no income. What help can I get?",
        user_info={"age": 67, "income": 60_000, "state": "bihar"},
    )

    print("\n===== RAG Agent Result =====")
    print(f"Recommended Schemes : {result['recommended_schemes']}")
    print(f"Answer              : {result['answer']}")
    print(f"Sources             : {result['sources']}")

