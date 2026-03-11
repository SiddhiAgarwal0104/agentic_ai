import pytest
from main_graph import bharat_sahayak
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main_graph import bharat_sahayak

test_cases = [
    {
        "input": "Mera naam Tanya hai aur main Kanpur se hoon, ration card issue hai.",
        "expected_intent": "Ration",
        "expected_name": "Tanya",
        "expected_location": "Kanpur"
    },
    {
        "input": "I need to visit the government hospital in Delhi.",
        "expected_intent": "Health",
        "expected_name": None,
        "expected_location": "Delhi"
    }
]

@pytest.mark.parametrize("case", test_cases)
def test_full_pipeline(case):
  
    result = bharat_sahayak.invoke({"user_input": case["input"]})
    issue = result["structured_issue"]
    
  
    assert issue["intent"] == case["expected_intent"]
    assert issue["entities"]["location"].lower() == case["expected_location"].lower()
    
  
    if case["expected_name"]:
        assert issue["entities"]["name"] == case["expected_name"]
    

    assert issue["confidence_score"] > 0.30 