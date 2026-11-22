import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "dataset" / "cleaned_customer_churn.csv"
)

CHART_DIR = BASE_DIR / "outputs" / "eda_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.savefig(CHART_DIR / "01_churn_distribution.png",
            bbox_inches="tight")

plt.show()


# Churn by Contract Type

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn",
    palette="Set1"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.savefig(CHART_DIR / "02_churn_by_contract.png",
            bbox_inches="tight")

plt.show()


# Monthly Charges vs Churn

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.savefig(CHART_DIR / "03_monthly_charges.png",
            bbox_inches="tight")

plt.show()



# Customer Tenure vs Churn

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True,
    multiple="stack"
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.savefig(CHART_DIR / "04_tenure_churn.png",
            bbox_inches="tight")

plt.show()



# Churn by Payment Method

plt.figure(figsize=(11, 6))

sns.countplot(
    data=df,
    y="PaymentMethod",
    hue="Churn",
    palette="Set2"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Number of Customers")
plt.ylabel("Payment Method")

plt.savefig(CHART_DIR / "05_payment_method.png",
            bbox_inches="tight")

plt.show()


# Churn by Internet Service

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn",
    palette="Set2"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.savefig(CHART_DIR / "06_internet_service.png",
            bbox_inches="tight")

plt.show()



# Churn Percentage by Contract

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract:")
print(contract_churn.round(2))


# Churn Percentage by Internet Service

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Internet Service:")
print(internet_churn.round(2))


# Overall Churn Percentage

overall_churn = (
    df["Churn"].eq("Yes").mean() * 100
)

print("\nOverall Churn Rate:")
print(round(overall_churn, 2), "%")
