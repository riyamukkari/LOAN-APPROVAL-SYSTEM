# Loan Approval Prediction Pipeline  
Machine Learning Classification with End-to-End Preprocessing and Hyperparameter Tuning

## Introduction

This project implements a complete machine learning pipeline for predicting loan approval status using structured applicant data. The workflow integrates preprocessing, feature engineering, model tuning, and evaluation into a clean and production-style pipeline using scikit-learn.

## Objective

The objective is to build a robust classification model that predicts whether a loan application will be approved or rejected. The project emphasizes:

- Handling missing values systematically  
- Proper encoding of categorical variables  
- Preventing data leakage using pipelines  
- Hyperparameter tuning using cross-validation  
- Evaluating model performance using ROC-AUC  

## Dataset

The dataset consists of applicant-level financial and demographic features such as:

- Gender, Married, Dependents  
- Education, Self_Employed  
- ApplicantIncome, CoapplicantIncome  
- LoanAmount, Loan_Amount_Term  
- Credit_History, Property_Area  

Target Variable:
- Y → Approved  
- N → Rejected  

## Data Preprocessing

A structured preprocessing pipeline was built using `ColumnTransformer`:

### Numerical Features
- Missing value imputation using median  
- Feature scaling using StandardScaler  

### Categorical Features
- Missing value imputation using most frequent value  
- One-Hot Encoding with unknown category handling  
- Drop-first encoding to avoid multicollinearity  

Additional steps:
- Label encoding of the target variable  
- Stratified train-test split (80/20)  

All preprocessing steps were integrated inside a unified sklearn Pipeline to prevent data leakage.

## Exploratory Data Analysis

Initial analysis included:

- Inspecting dataset shape and structure  
- Reviewing sample records  
- Identifying missing values  
- Understanding feature types (numerical vs categorical)  

This ensured correct preprocessing strategy before model training.

## Model Building

### Algorithm Used
Random Forest Classifier

### Hyperparameter Tuning
GridSearchCV with:
- 5-fold cross-validation  
- ROC-AUC as scoring metric  
- Search across:
  - n_estimators  
  - max_depth  
  - min_samples_split  
  - class_weight  

The final model was selected based on best cross-validation ROC-AUC.

## Evaluation Metrics

- Accuracy  
- ROC-AUC Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  
- ROC Curve Visualization  

The best model achieved perfect performance on the test split in this synthetic dataset scenario.

## Conclusion

This project demonstrates how to build a clean, scalable, and production-ready machine learning classification pipeline. By integrating preprocessing, feature transformation, hyperparameter tuning, and evaluation into a single workflow, the model remains reproducible and robust. The structured pipeline design makes this approach suitable for real-world financial approval systems.
