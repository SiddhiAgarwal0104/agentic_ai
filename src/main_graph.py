from typing import TypedDict, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from preprocess import preprocessing_node
from loader import predict_intent, load_intent_model
from extractor import extract_entities
from rag_agent import recommend_schemes, initialise as init_rag_agent


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Member 1 fields (unchanged)
    user_input:       str
    clean_text:       str
    structured_issue: Dict

    # Member 2 fields (RAG Agent)
    user_info:              Dict          # age, income, state — passed in at runtime
    recommended_schemes:    List[str]     # eligible scheme categories
    answer:                 str           # Gemini-generated grounded answer
    sources:                List[str]     # PDF filenames used as context


# ── One-time startup ──────────────────────────────────────────────────────────
tokenizer, model = load_intent_model()
init_rag_agent()   # builds FAISS index + loads eligibility model (runs once)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def intent_node(state: AgentState) -> Dict:
    """Member 1 — preprocess + intent classification."""
    cleaned    = preprocessing_node(state["user_input"])
    prediction = predict_intent(cleaned, tokenizer, model)
    return {
        "clean_text":       cleaned,
        "structured_issue": prediction,
    }


def ner_node(state: AgentState) -> Dict:
    """Member 1 — named entity recognition."""
    entities  = extract_entities(state["clean_text"])
    new_issue = state["structured_issue"].copy()
    new_issue["entities"] = entities
    return {"structured_issue": new_issue}


def rag_node(state: AgentState) -> Dict:
    """
    Member 2 — Scheme Recommendation & RAG Agent.

    Uses:
      - state["clean_text"]  as the search query
      - state["user_info"]   for eligibility checking (age, income, state)

    Writes:
      - recommended_schemes, answer, sources
    """
    query     = state.get("clean_text", state["user_input"])
    user_info = state.get("user_info", {"age": 30, "income": 300_000, "state": "other"})

    result = recommend_schemes(query=query, user_info=user_info)

    return {
        "recommended_schemes": result["recommended_schemes"],
        "answer":              result["answer"],
        "sources":             result["sources"],
    }


# ── Graph ─────────────────────────────────────────────────────────────────────

builder = StateGraph(AgentState)

builder.add_node("classifier", intent_node)   # Member 1
builder.add_node("extractor",  ner_node)       # Member 1
builder.add_node("rag",        rag_node)       # Member 2

builder.add_edge(START,        "classifier")
builder.add_edge("classifier", "extractor")
builder.add_edge("extractor",  "rag")          # RAG runs after NER
builder.add_edge("rag",        END)

bharat_sahayak = builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_data = {
        "user_input": "Mera naam Tanya hai aur main Kanpur se hoon, mera ration card nahi mila.",
        "user_info":  {"age": 34, "income": 120_000, "state": "uttar pradesh"},
    }

    final_state = bharat_sahayak.invoke(input_data)

    print("\n===== Bharat Sahayak AI — Final State =====")
    print(f"Structured Issue    : {final_state['structured_issue']}")
    print(f"Recommended Schemes : {final_state['recommended_schemes']}")
    print(f"Answer              : {final_state['answer']}")
    print(f"Sources             : {final_state['sources']}")