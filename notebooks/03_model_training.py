import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "dataset" / "cleaned_customer_churn.csv"
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

#data preprocess

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Remove customerID because it does not help predict churn
df = df.drop(columns=["customerID"])

# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify numerical,categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object", "bool", "category"]).columns

print("\nNumerical Columns:", list(numerical_columns))
print("\nCategorical Columns:", list(categorical_columns))

# Preprocess numerical , categorical data
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_columns),
    ("cat", categorical_transformer, categorical_columns)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nPreprocessing setup completed successfully!")

#model trainig

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic pipeline
logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Random forest pipeline
random_forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train Logistic 
print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)
print("Logistic Regression training completed!")

# Train Rf
print("\nTraining Random Forest...")
random_forest_model.fit(X_train, y_train)
print("Random Forest training completed!")

print("\nBoth models trained successfully!")

#model evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import pandas as pd

# Store trained models
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

results = []

for name, model in models.items():

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of churn (class 1)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    print(f"\n{'=' * 45}")
    print(f"MODEL: {name}")
    print("=" * 45)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Compare both models
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# Save comparison results
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

results_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

#best model rf

import joblib

# Select Random Forest as the chosen model
best_model = random_forest_model

# Create models folder if it does not exist
models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)

# Save trained model
model_path = models_dir / "best_churn_model.pkl"

joblib.dump(best_model, model_path)

print("\nBest model saved successfully!")
print("Model location:", model_path)

#feature analysis

import matplotlib.pyplot as plt
import pandas as pd

# Get the trained Random Forest classifier
rf_classifier = best_model.named_steps["classifier"]

# Get transformed feature names from preprocessing
feature_names = best_model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Extract feature importance values
importance_values = rf_classifier.feature_importances_

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# Sort features by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display top 15 important features
print("\nTOP 15 IMPORTANT FEATURES")
print(feature_importance.head(15).to_string(index=False))

# Plot top 15 features
top_features = feature_importance.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 15 Factors Influencing Customer Churn")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()

# Save chart
chart_path = BASE_DIR / "outputs" / "feature_importance.png"
chart_path.parent.mkdir(exist_ok=True)

plt.savefig(chart_path, dpi=300)
plt.show()

# Save feature importance data
feature_importance.to_csv(
    BASE_DIR / "outputs" / "feature_importance.csv",
    index=False
)

print("\nFeature importance analysis completed!")



# ==========================================
# STEP 12: MODEL EVALUATION VISUALIZATIONS
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)

# Make predictions using the best model
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Create output folder
chart_dir = BASE_DIR / "outputs"
chart_dir.mkdir(exist_ok=True)

# ------------------------------------------
# 1. CONFUSION MATRIX HEATMAP
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    chart_dir / "confusion_matrix.png",
    dpi=300
)

plt.show()

# ------------------------------------------
# 2. ROC CURVE
# ------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guess"
)

plt.title("ROC Curve - Customer Churn Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    chart_dir / "roc_curve.png",
    dpi=300
)

plt.show()

print("\nEvaluation visualizations saved successfully!")
