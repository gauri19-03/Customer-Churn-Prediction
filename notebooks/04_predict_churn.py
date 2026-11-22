import joblib
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model
model_path = BASE_DIR / "models" / "best_churn_model.pkl"
model = joblib.load(model_path)

print("Customer Churn Prediction System")
print("-" * 40)

# Sample new customer information
new_customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 427.50
}])

# Predict churn
prediction = model.predict(new_customer)[0]

# Get churn probability
probability = model.predict_proba(new_customer)[0][1]

# Display results
print("\nCustomer Information:")
print(new_customer.to_string(index=False))

print("\nPrediction Result:")

if prediction == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer is likely to STAY.")

print(f"\nChurn Probability: {probability * 100:.2f}%")

print("\nPrediction completed successfully!")
