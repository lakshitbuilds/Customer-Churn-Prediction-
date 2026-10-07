# Customer Churn Prediction

A beginner-friendly Machine Learning project that predicts whether a telecom customer is likely to leave a company.

## What this project does

The project uses customer information such as tenure, contract type, monthly charges, payment method, and support calls to predict churn.

Two classification models are compared:

- Logistic Regression
- Decision Tree

## Tech Stack

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

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

## Notebook Workflow

1. Load the dataset
2. Check columns, data types, missing values, and duplicates
3. Prepare the data
4. Perform basic EDA
5. Encode categorical columns
6. Split training and testing data
7. Scale numerical features
8. Train Logistic Regression
9. Evaluate using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix
10. Compare with Decision Tree
11. Test the model with a new customer

## Run the project

Clone the repository:

```bash
git clone https://github.com/lakshitbuilds/Customer-Churn-Prediction-.git
cd Customer-Churn-Prediction-
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start Jupyter Lab:

```bash
jupyter lab
```

Open:

```
notebooks/customer_churn_analysis.ipynb
```

You can also run the Python file:

```bash
python src/train_model.py
```

## Dataset

The repository currently contains a small synthetic telecom churn dataset for learning and practice. Because the dataset is small, model scores should only be treated as demonstration results.

## Author

Lakshit Suthar
