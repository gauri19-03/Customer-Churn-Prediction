import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
df = pd.read_csv(
    BASE_DIR / "dataset" / "cleaned_customer_churn.csv"
)

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Same test split used in previous steps
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load both models
original_model = joblib.load(
    BASE_DIR / "models" / "best_churn_model.pkl"
)

tuned_model = joblib.load(
    BASE_DIR / "models" / "tuned_churn_model.pkl"
)

# Evaluate models
models = {
    "Original Random Forest": original_model,
    "Tuned Random Forest": tuned_model
}

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    })

# Create comparison table
comparison_df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(comparison_df.round(4).to_string(index=False))

# Save report
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

comparison_df.to_csv(
    output_dir / "final_model_comparison.csv",
    index=False
)

print("\nFinal comparison report saved successfully!")
