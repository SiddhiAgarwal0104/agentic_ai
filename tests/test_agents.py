import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from main_graph import bharat_sahayak


test_cases = [
    {
        "input": "Namaste, mera naam Rajesh Kumar hai aur main Lucknow se hoon. Mujhe naya ration card chahiye.",
        "expected_intent": "ration",
        "expected_name": "Rajesh Kumar",
        "expected_location": "Lucknow"
    },
    {
        "input": "My father is a retired teacher in Patna, his pension has not been credited this month.",
        "expected_intent": "pension",
        "expected_name": "Unknown", 
        "expected_location": "Patna"
    },
    {
        "input": "Urgent help needed at the civil hospital in Mumbai for a viral fever patient.",
        "expected_intent": "health",
        "expected_name": "Unknown",
        "expected_location": "Mumbai"
    },
    {
        "input": "Sunita Devi here from Bhopal. Mera pension card list mein naam nahi mil raha hai.",
        "expected_intent": "pension",
        "expected_name": "Sunita Devi",
        "expected_location": "Bhopal"
    }
]

@pytest.mark.parametrize("case", test_cases)
def test_full_pipeline(case):

    result = bharat_sahayak.invoke({"user_input": case["input"]})
    issue = result["structured_issue"]
    
 
    predicted_intent = str(issue["intent"]).strip().lower()
    expected_intent = case["expected_intent"].strip().lower()
    
    assert predicted_intent == expected_intent, f"Intent Mismatch: Expected {expected_intent} but got {predicted_intent}"

    assert issue["confidence_score"] > 0.40, f" Confidence score too low: {issue['confidence_score']}"


    if case["expected_location"] != "Unknown":
        actual_loc = issue["entities"].get("location")
        assert actual_loc is not None, "Location was not extracted"
        assert actual_loc.lower() == case["expected_location"].lower()
    

    if case["expected_name"] != "Unknown":
        actual_name = issue["entities"].get("name")
        assert actual_name == case["expected_name"], f" Name Mismatch: Expected {case['expected_name']} but got {actual_name}"