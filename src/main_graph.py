from typing import TypedDict, Dict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    user_input: str
    clean_text: str
    structured_issue: Dict
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    user_input: str
    clean_text: str
    structured_issue: Dict
from preprocess import preprocessing_node
from loader import predict_intent, load_intent_model
from extractor import extract_entities


tokenizer, model = load_intent_model()

def intent_node(state: AgentState):

    cleaned = preprocessing_node(state['user_input'])

    prediction = predict_intent(cleaned, tokenizer, model)
    return {"clean_text": cleaned, "structured_issue": prediction}

def ner_node(state: AgentState):
   
    entities = extract_entities(state['clean_text'])
    
  
    new_issue = state['structured_issue'].copy()
    new_issue['entities'] = entities
    
    return {"structured_issue": new_issue}

builder = StateGraph(AgentState)


builder.add_node("classifier", intent_node)
builder.add_node("extractor", ner_node)


builder.add_edge(START, "classifier")
builder.add_edge("classifier", "extractor") 
builder.add_edge("extractor", END) 


bharat_sahayak = builder.compile()

input_data = {"user_input": "Mera naam Tanya hai aur main Kanpur se hoon, mera ration card nahi mila."}
final_state = bharat_sahayak.invoke(input_data)

print(final_state['structured_issue'])