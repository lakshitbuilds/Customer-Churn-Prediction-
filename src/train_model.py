from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def show_metrics(name, y_test, predictions):
    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, predictions), 3))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 3))
    print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 3))
    print("F1 Score :", round(f1_score(y_test, predictions, zero_division=0), 3))


data_path = Path(__file__).resolve().parents[1] / "data" / "customer_churn.csv"

df = pd.read_csv(data_path)
df = df.drop(columns="CustomerID")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns="Churn")
y = df["Churn"]

X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

scaler = StandardScaler()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

print("Customer Churn Prediction")
show_metrics("Logistic Regression", y_test, log_pred)
show_metrics("Decision Tree", y_test, tree_pred)
