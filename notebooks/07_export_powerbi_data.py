import pandas as pd
import joblib
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load cleaned dataset
df = pd.read_csv(
    BASE_DIR / "dataset" / "cleaned_customer_churn.csv"
)

# Load selected original model
model = joblib.load(
    BASE_DIR / "models" / "best_churn_model.pkl"
)

# Prepare customer features
X = df.drop(columns=["customerID", "Churn"])

# Generate predictions
df["Predicted_Churn"] = model.predict(X)
df["Churn_Probability"] = model.predict_proba(X)[:, 1]

# Convert prediction values into labels
df["Predicted_Churn"] = df["Predicted_Churn"].map({
    0: "No",
    1: "Yes"
})

# Export Power BI dataset
output_dir = BASE_DIR / "dashboard"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "customer_churn_powerbi.csv"

df.to_csv(output_path, index=False)

print("\nPower BI dataset exported successfully!")
print("File location:", output_path)

print("\nDataset shape:", df.shape)
print("\nPredicted churn distribution:")
print(df["Predicted_Churn"].value_counts())
