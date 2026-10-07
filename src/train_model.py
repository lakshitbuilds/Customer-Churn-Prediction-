from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(name, y_true, y_pred):
    """Print basic classification metrics."""
    print(f"\n{name}")
    print("-" * len(name))
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 3))
    print("Recall   :", round(recall_score(y_true, y_pred, zero_division=0), 3))
    print("F1 Score :", round(f1_score(y_true, y_pred, zero_division=0), 3))


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "customer_churn.csv"

    # 1. Load data
    df = pd.read_csv(data_path)

    # 2. Remove ID column and encode target
    df = df.drop(columns=["CustomerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # 3. Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 4. Encode categorical features
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    # 5. Split before scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 6. Scale numerical features for Logistic Regression
    scale_columns = ["Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaler = StandardScaler()
    X_train_scaled[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test_scaled[scale_columns] = scaler.transform(X_test[scale_columns])

    # 7. Logistic Regression
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)
    logistic_predictions = logistic_model.predict(X_test_scaled)

    # 8. Decision Tree - scaling is not required
    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)

    # 9. Evaluation
    print("Customer Churn Prediction")
    print("=========================")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    evaluate_model("Logistic Regression", y_test, logistic_predictions)
    evaluate_model("Decision Tree", y_test, tree_predictions)


if __name__ == "__main__":
    main()
