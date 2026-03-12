import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from monitoring import predict_resolution, recommend_escalation


def test_monitoring():

    intent = "education"
    urgency = "low"

    resolution_time = predict_resolution(intent, urgency)
    escalation = recommend_escalation(intent, urgency)

    print("Intent:", intent)
    print("Urgency:", urgency)
    print("Predicted Resolution Time:", resolution_time, "hours")
    print("Escalation Decision:", escalation)


if __name__ == "__main__":
    test_monitoring()