from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END

from speech_to_text import record_audio, transcribe_audio
from text_to_speech import speak_text
from monitoring_agent import predict_resolution, recommend_escalation 

from preprocess import preprocessing_node
from loader import predict_intent, load_intent_model
from extractor import extract_entities

from rag_agent import recommend_schemes, initialise as init_rag_agent

# ── State ─────────────────────────────────────────────
class AgentState(TypedDict):
    user_input: str
    clean_text: str
    structured_issue: Dict
    user_info: Dict
    recommended_schemes: List[str]
    answer: str
    expected_resolution_time: float
    escalation_status: str

# ── Startup ───────────────────────────────────────────
tokenizer, model = load_intent_model()
init_rag_agent()

# ── Nodes ─────────────────────────────────────────────

def intent_node(state: AgentState) -> Dict:
    cleaned = preprocessing_node(state["user_input"])
    prediction = predict_intent(cleaned, tokenizer, model)
    return {
        "clean_text": cleaned,
        "structured_issue": prediction,
    }

def ner_node(state: AgentState) -> Dict:
    entities = extract_entities(state["clean_text"])
    issue = state["structured_issue"].copy()
    issue["entities"] = entities
    return {"structured_issue": issue}

def rag_node(state: AgentState) -> Dict:
    query = state.get("clean_text", state["user_input"])
    user_info = state["user_info"]
    result = recommend_schemes(query=query, user_info=user_info)
    return {
        "recommended_schemes": result["recommended_schemes"],
        "answer": result["answer"],
        "sources": result["sources"],
    }

def monitoring_node(state: AgentState) -> Dict:
    intent = state["structured_issue"].get("intent", "general")
    urgency = state["structured_issue"].get("urgency", "low")
    
    res_time = predict_resolution(intent, urgency)
    status = recommend_escalation(intent, urgency)
    
    return {
        "expected_resolution_time": res_time,
        "escalation_status": status
    }

# ── Graph ─────────────────────────────────────────────

builder = StateGraph(AgentState)

# Add all 4 nodes
builder.add_node("classifier", intent_node)
builder.add_node("extractor", ner_node)
builder.add_node("rag", rag_node)
builder.add_node("monitor", monitoring_node) # Monitoring

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "extractor")
builder.add_edge("extractor", "rag")    # Link Member 1 to Member 2
builder.add_edge("rag", "monitor")      # Link Member 2 to Member 4
builder.add_edge("monitor", END)        # Final step

bharat_sahayak = builder.compile()

# ── Run System ─────────────────────────────────────────

if __name__ == "__main__":

    # 🎤 Step 1: Record Audio 
    print("Listening... speak now.")
    audio_file = record_audio()

    # 🧠 Step 2: Speech → Text 
    user_query = transcribe_audio(audio_file)
    print("\nUser said:", user_query)

    # Initialize data
    input_data = {
        "user_input": user_query,
        "user_info": {"age": 34, "income": 120000, "state": "uttar pradesh"},
    }

    # 🤖 Step 3: Run AI Agents (The Graph)
    final_state = bharat_sahayak.invoke(input_data)

    print("\n===== Bharat Sahayak AI — Final State =====")
    print("Intent:", final_state["structured_issue"].get("intent"))
    print("Resolution Time:", final_state["expected_resolution_time"], "hours")
    print("Escalation:", final_state["escalation_status"])
    print("Answer:", final_state["answer"])

    # 🔊 Step 4: Speak the Answer
    final_voice_output = f"{final_state['answer']}. Your case status is: {final_state['escalation_status']}."
    speak_text(final_voice_output)