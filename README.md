# Credit Card Fraud Detection

## Project Overview

A machine learning based Credit Card Fraud Detection system that predicts whether a transaction is fraudulent or legitimate.

The project handles class imbalance using SMOTE and compares Random Forest and XGBoost models.

## Objectives

- Detect fraudulent credit card transactions.
- Handle class imbalance using SMOTE.
- Compare Random Forest and XGBoost.
- Perform hyperparameter tuning.
- Optimize the classification threshold.
- Evaluate models using classification metrics.
- Package the final model for reuse and deployment.

## Dataset Features

The dataset contains:

- transaction_id
- amount
- transaction_hour
- merchant_category
- foreign_transaction
- location_mismatch
- device_trust_score
- velocity_last_24h
- cardholder_age
- is_fraud

Target column:

`is_fraud`

## Machine Learning Workflow
```
Dataset
↓
Data Preprocessing
↓
Categorical Encoding
↓
Train/Test Split
↓
SMOTE
↓
Random Forest / XGBoost
↓
Hyperparameter Tuning
↓
Threshold Tuning
↓
Model Evaluation
↓
Model Packaging
```
## Models Used

### Random Forest

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees for classification.

### XGBoost

XGBoost is a gradient boosting algorithm used for classification and is used as the final model in this project.

## Handling Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) is used to generate synthetic samples for the minority fraud class.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- PR-AUC

## Final Model

The final deployment model is a tuned XGBoost pipeline.

The packaged model is stored as:

`models/fraud_detection_model.pkl`

## Project Structure

```text
credit-card-fraud-detector/
│
├── models/
│   └── fraud_detection_model.pkl
│
├── notebooks/
│   └── EDPProject.ipynb
│
├── results/
│   ├── threshold_analysis.csv
│   └── model_comparison.csv
│
├── src/
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

pip install -r requirements.txt

## Prediction

The packaged model can be used through:

src/predict.py

Run:

python src/predict.py

## Model Packaging

The final trained pipeline is saved using Joblib.

The saved model can be loaded without retraining the machine learning model.

## Deployment Preparation

The project is prepared for deployment by:

- Packaging the final trained model.
- Separating prediction code from the training notebook.
- Providing requirements.txt.
- Providing .gitignore.
- Organizing the project into a GitHub-ready structure.
- Creating a reusable prediction script.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Joblib

## Author
G. Yuva Sai -  24951A12C7
