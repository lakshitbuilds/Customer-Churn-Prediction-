# Customer Churn Prediction

A beginner-friendly Data Science and Machine Learning portfolio project that predicts whether a telecom customer is likely to churn.

## Current Progress

**Step 1 — Dataset Understanding**

The target variable is **Churn** (`Yes` = customer left, `No` = customer stayed).

## Project Structure

```
Customer-Churn-Prediction-/
├── data/
│   └── customer_churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Step 1 concepts

The notebook demonstrates `pd.read_csv()`, `head()`, `shape`, `columns`, `dtypes`, `info()`, `describe()`, and `value_counts()`.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/customer_churn_analysis.ipynb` and run cells top to bottom.

> The included dataset is a small synthetic telecom dataset created for learning. It is not real customer data.
