import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)


print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()


print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values After Conversion:")
print(df.isnull().sum())


print("\nMissing TotalCharges Details:")
print(df.loc[
    df["TotalCharges"].isnull(),
    ["tenure", "MonthlyCharges", "TotalCharges"]
])


df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nDuplicate Records:")
print(df.duplicated().sum())

df = df.drop_duplicates()


print("\nUpdated Data Types:")
print(df.dtypes)


print("\nChurn Distribution:")
print(df["Churn"].value_counts())

output_path = BASE_DIR / "dataset" / "cleaned_customer_churn.csv"

df.to_csv(output_path, index=False)

print("\nData Cleaning Completed!")
print("Final Shape:", df.shape)
print("Cleaned Dataset Saved At:", output_path)
