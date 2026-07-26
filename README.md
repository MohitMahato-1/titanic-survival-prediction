Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using the classic Kaggle Titanic dataset. Built as a hands-on exercise to practice the full ML workflow — data cleaning, feature engineering, and model building.

Project Overview

This project explores the Titanic dataset to understand which factors most influenced passenger survival, then engineers features and trains classification models to predict survival outcomes for unseen passengers.

Dataset

The data comes from the Kaggle Titanic competition and includes:

data/train.csv — training data with known survival outcomes
data/test.csv — test data for generating predictions
data/gender_submission.csv — sample submission format
Project Structure
Titanic-survival-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebook/
│   └── titanic_analysis.ipynb
└── README.md
Approach
Exploratory Data Analysis (EDA) — understanding distributions, missing values, and relationships between features and survival.
Feature Engineering:
FamilySize — combining SibSp and Parch to capture total family members aboard
IsAlone — flag for passengers traveling without family
Title — extracted from passenger names (Mr, Mrs, Miss, Master, etc.) as a proxy for social status and age group
Model Building (in progress) — training and evaluating classification models (e.g., Logistic Regression, Random Forest, KNN) to predict survival.
Evaluation — using metrics like accuracy, precision, recall, and F1 score to assess model performance.
Tools & Libraries
Python
pandas, NumPy
scikit-learn
matplotlib / seaborn (for visualization)
Jupyter Notebook
Status

🚧 Work in progress — feature engineering complete, model building and evaluation next.

Author

Mohit Mahato — GitHub