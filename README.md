# Customer Churn Prediction

A beginner-friendly end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave a company.

## Project Objective

The goal is to predict **Customer Churn**:

- **Churn = Yes** → customer is likely to leave
- **Churn = No** → customer is likely to stay

This project is designed for learning, resume/GitHub use, and fresher-level interview explanation.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```
Customer-Churn-Prediction-/
├── data/
│   └── customer_churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   └── train_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine Learning Workflow

The notebook covers the complete basic ML pipeline:

1. Load and understand the dataset
2. Check shape, columns, data types, and summary
3. Check missing values and duplicate rows
4. Remove the unnecessary CustomerID column
5. Perform Exploratory Data Analysis (EDA)
6. Convert categorical features into numerical features
7. Split data into training and testing sets
8. Scale numerical features correctly
9. Train a Logistic Regression model
10. Evaluate using:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
11. Compare Logistic Regression with Decision Tree
12. Predict churn for a new customer

## Important ML Concept Used

The project performs the train/test split **before scaling**.

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Why?

- `fit_transform()` learns mean and standard deviation from the training data.
- `transform()` applies the same values to test data.
- This prevents **data leakage**.

Decision Tree is trained without scaling because tree-based models do not depend on feature magnitude.

## Models

### Logistic Regression
Used as the main classification model because it is simple, interpretable, and suitable for binary classification.

### Decision Tree
Used as a comparison model.

## Dataset

The included dataset contains **40 synthetic telecom customer records**.

Important columns include:

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- InternetService
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges
- SupportCalls
- Churn

> This dataset is intentionally small and synthetic for learning. Very high scores on it should not be treated as proof of production-level model performance.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/lakshitbuilds/Customer-Churn-Prediction-.git
cd Customer-Churn-Prediction-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```
notebooks/customer_churn_analysis.ipynb
```

### 4. Or run the Python training script

```bash
python src/train_model.py
```

## Evaluation Metrics

- **Accuracy**: overall percentage of correct predictions
- **Precision**: out of predicted churn customers, how many actually churned
- **Recall**: out of actual churn customers, how many the model correctly found
- **F1 Score**: balance between precision and recall

For churn prediction, **Recall is especially important** because a false negative means the company failed to identify a customer who was actually going to leave.

## Future Improvements

Possible next steps:

- Use a larger real-world telecom churn dataset
- Add more features
- Perform feature importance analysis
- Compare additional classification algorithms
- Tune model hyperparameters
- Build a simple prediction web app

## Author

**Lakshit Suthar**
