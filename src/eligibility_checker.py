"""
eligibility_checker.py
----------------------
Rule-based + ML eligibility checker.

Uses a Decision Tree trained on a synthetic dataset that maps:
    (age, income_inr, state_encoded) → [list of eligible scheme categories]

In production, replace the synthetic data with a real labeled dataset
(see data/eligibility/ — CSV with one row per citizen profile).
"""
from dotenv import load_dotenv
load_dotenv()
import os
import pickle
import numpy as np
from typing import Dict, List, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

# Path to persist the trained model
MODEL_PATH = os.getenv("ELIGIBILITY_MODEL_PATH", "data/models/eligibility_tree.pkl")

# ── State encoding ────────────────────────────────────────────────────────────
# Map state names to integers (add more states as needed)
STATE_MAP: Dict[str, int] = {
    "andhra pradesh": 1, "arunachal pradesh": 2, "assam": 3, "bihar": 4,
    "chhattisgarh": 5, "goa": 6, "gujarat": 7, "haryana": 8,
    "himachal pradesh": 9, "jharkhand": 10, "karnataka": 11, "kerala": 12,
    "madhya pradesh": 13, "maharashtra": 14, "manipur": 15, "meghalaya": 16,
    "mizoram": 17, "nagaland": 18, "odisha": 19, "punjab": 20,
    "rajasthan": 21, "sikkim": 22, "tamil nadu": 23, "telangana": 24,
    "tripura": 25, "uttar pradesh": 26, "uttarakhand": 27, "west bengal": 28,
    "delhi": 29, "other": 0,
}

# Scheme categories (one binary output per category)
SCHEME_CATEGORIES = [
    "PM_Kisan",           # Farmer income support (age 18+, income < 2L)
    "PM_Awas_Yojana",     # Housing (income < 3L)
    "Ayushman_Bharat",    # Health insurance (income < 5L)
    "PM_Jan_Dhan",        # Banking / financial inclusion (all)
    "Sukanya_Samriddhi",  # Girl child savings (age < 10, proxy: guardian age 18-45)
    "Senior_Pension",     # Old age pension (age >= 60, income < 1L)
    "PM_Scholarship",     # Education scholarship (age 18-25, income < 2.5L)
    "MGNREGA",            # Rural employment (age 18+, rural states)
]


# ── Synthetic training data ───────────────────────────────────────────────────

def _make_synthetic_dataset(n_samples: int = 1000):
    """
    Generate a synthetic (age, income, state) → scheme eligibility dataset.
    Replace this with real data from data/eligibility/profiles.csv in production.
    """
    rng = np.random.default_rng(42)

    ages    = rng.integers(18, 80, size=n_samples)
    incomes = rng.integers(50_000, 1_500_000, size=n_samples)   # INR per year
    states  = rng.integers(0, 30, size=n_samples)

    labels = np.zeros((n_samples, len(SCHEME_CATEGORIES)), dtype=int)

    for i in range(n_samples):
        a, inc, s = int(ages[i]), int(incomes[i]), int(states[i])

        labels[i, 0] = int(a >= 18 and inc < 200_000)               # PM_Kisan
        labels[i, 1] = int(inc < 300_000)                           # PM_Awas
        labels[i, 2] = int(inc < 500_000)                           # Ayushman
        labels[i, 3] = 1                                            # Jan Dhan (all)
        labels[i, 4] = int(18 <= a <= 45)                           # Sukanya (guardian)
        labels[i, 5] = int(a >= 60 and inc < 100_000)               # Senior pension
        labels[i, 6] = int(18 <= a <= 25 and inc < 250_000)         # Scholarship
        labels[i, 7] = int(a >= 18 and s in range(3, 20))           # MGNREGA (rural)

    X = np.column_stack([ages, incomes, states])
    return X, labels


# ── Model training ────────────────────────────────────────────────────────────

def train_eligibility_model() -> MultiOutputClassifier:
    """
    Train a multi-output Decision Tree on synthetic data and save to disk.

    Returns:
        Trained MultiOutputClassifier wrapping DecisionTreeClassifier.
    """
    print("[EligibilityChecker] Training Decision Tree on synthetic data...")
    X, y = _make_synthetic_dataset(n_samples=2000)

    clf = MultiOutputClassifier(
        DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    )
    clf.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"[EligibilityChecker] Model saved → {MODEL_PATH}")
    return clf


# Cache trained model in memory
_clf = None


def _get_model() -> MultiOutputClassifier:
    """Load or train the eligibility model (singleton)."""
    global _clf
    if _clf is not None:
        return _clf

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _clf = pickle.load(f)
        print("[EligibilityChecker] Loaded existing model.")
    else:
        _clf = train_eligibility_model()

    return _clf


# ── Public API ────────────────────────────────────────────────────────────────

def check_eligibility(user_info: Dict[str, Any]) -> List[str]:
    """
    Predict which scheme categories a user is eligible for.

    Args:
        user_info: Dict with keys:
            "age"    (int)   – user's age in years
            "income" (float) – annual income in INR
            "state"  (str)   – state name (lowercase)

    Returns:
        List of eligible scheme category names.

    Example:
        >>> check_eligibility({"age": 65, "income": 80000, "state": "bihar"})
        ["Ayushman_Bharat", "PM_Jan_Dhan", "Senior_Pension", "MGNREGA"]
    """
    age    = int(user_info.get("age", 30))
    income = float(user_info.get("income", 300_000))
    state  = str(user_info.get("state", "other")).lower().strip()
    state_enc = STATE_MAP.get(state, 0)

    X = np.array([[age, income, state_enc]])
    model = _get_model()
    preds = model.predict(X)[0]   # shape: (n_schemes,)

    eligible = [
        SCHEME_CATEGORIES[i]
        for i, flag in enumerate(preds)
        if flag == 1
    ]
    return eligible