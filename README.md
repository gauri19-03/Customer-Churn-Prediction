# Customer Churn Prediction & Analysis

**Tools:** Python, Pandas, NumPy, Scikit-learn, SQL, Power BI  
**Project:** Customer Churn Prediction & Analysis  
**Date:** November 2026

## Project Overview

This project analyzes customer churn in a telecommunications company and builds machine learning models to predict which customers are likely to leave.

The goal is to identify customer churn patterns, understand important factors influencing churn, and provide insights that can help businesses improve customer retention.

## Objectives

- Analyze customer behavior and churn patterns.
- Clean and preprocess customer data.
- Perform exploratory data analysis (EDA).
- Use SQL to investigate churn trends.
- Build and evaluate machine learning models.
- Tune model hyperparameters.
- Create an interactive Power BI dashboard.

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains customer information such as:

- Customer demographics
- Account information
- Services subscribed to
- Contract and payment details
- Monthly and total charges
- Customer churn status

Source: [IBM Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and machine learning |
| Pandas & NumPy | Data cleaning and manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning and model evaluation |
| SQL | Customer churn analysis |
| Power BI | Interactive dashboard |
| Joblib | Model serialization |

## Project Workflow

### 1. Data Cleaning
- Handled missing values.
- Converted `TotalCharges` into numeric format.
- Removed duplicate records.
- Exported the cleaned dataset.

### 2. Exploratory Data Analysis
- Analyzed customer churn distribution.
- Studied churn across contract types.
- Compared churn across internet services.
- Explored customer tenure and payment methods.
- Created visualizations to identify churn patterns.

### 3. SQL Analysis
Performed SQL queries to analyze:
- Total customer count.
- Churned customer count and churn rate.
- Churn by contract type.
- Churn by payment method.
- Average monthly charges by churn status.
- Churn by internet service.

### 4. Machine Learning

Built and evaluated two classification models:

- Logistic Regression
- Random Forest Classifier

Used a Scikit-learn preprocessing pipeline with numerical scaling and categorical encoding.

### 5. Model Evaluation

Initial model comparison:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.21% |
| Random Forest | 75.59% | 52.83% | 74.87% | 61.95% | 83.79% |

Random Forest achieved higher recall and F1 score, making it a useful choice when identifying as many potentially churning customers as possible is important.

Hyperparameter tuning was also performed using GridSearchCV.

### 6. Power BI Dashboard

Created an interactive dashboard containing:

- Total Customers
- Churned Customers
- Churn Rate
- Average Monthly Charges
- Customer Churn Distribution
- Churn Rate by Contract
- Churn Rate by Internet Service
- Churn Rate by Payment Method
- Churn Rate by Tenure
- Interactive slicers for Contract, Gender, and Internet Service

## Project Structure

```text
Customer-Churn-Prediction/
│
├── dataset/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── cleaned_customer_churn.csv
│
├── notebooks/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_model_training.py
│   ├── 04_predict_churn.py
│   ├── 05_model_tuning.py
│   ├── 06_final_model_comparison.py
│   └── 07_export_powerbi_data.py
│
├── sql/
│   └── churn_analysis.sql
│
├── models/
│   ├── best_churn_model.pkl
│   └── tuned_churn_model.pkl
│
├── dashboard/
│   ├── customer_churn_powerbi.csv
│   └── Customer_Churn_Dashboard.pbix
│
├── outputs/
│
└── requirements.txt
```

## Key Insights

- Customers on month-to-month contracts show higher churn.
- Contract type is an important factor in customer retention.
- Internet service type is associated with differences in churn.
- Customer tenure and payment methods provide useful information for churn analysis.

## How to Run the Project

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd Customer-Churn-Prediction
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Python scripts in workflow order:

```bash
python -m notebooks.01_data_cleaning
python -m notebooks.02_eda
python -m notebooks.03_model_training
```

Continue with the remaining scripts as needed.

## Conclusion

This project demonstrates an end-to-end customer churn analysis workflow, combining data analytics, SQL, machine learning, and business intelligence.

It demonstrates how customer data can be transformed into predictive insights that support better customer retention strategies.

---

 
**Skills Demonstrated:** Data Analytics | Python | SQL | Machine Learning | Power BI
