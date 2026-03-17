from typing import TypedDict, Dict
from langgraph.graph import StateGraph, START, END
from preprocess import preprocessing_node
from loader import load_intent_model, predict_intent
from extractor import extract_entities

class AgentState(TypedDict):
    user_input: str
    clean_text: str
    structured_issue: Dict

tokenizer, model = load_intent_model()

def classification_node(state: AgentState):
    
    prediction = predict_intent(state["clean_text"], tokenizer, model)
    return {"structured_issue": prediction}

def extraction_node(state: AgentState):
    entities = extract_entities(state["clean_text"])
    
 
    updated_issue = state["structured_issue"].copy()
    updated_issue["entities"] = entities
    return {"structured_issue": updated_issue}

builder = StateGraph(AgentState)
builder.add_node("cleaner", preprocessing_node)
builder.add_node("classifier", classification_node)
builder.add_node("extractor", extraction_node)

builder.add_edge(START, "cleaner")
builder.add_edge("cleaner", "classifier")
builder.add_edge("classifier", "extractor")
builder.add_edge("extractor", END)

bharat_sahayak = builder.compile()