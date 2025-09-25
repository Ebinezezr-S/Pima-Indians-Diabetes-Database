🩺 Pima Indians Diabetes Prediction
📝 Project Overview

This project uses the Pima Indians Diabetes Database to predict whether a patient is likely to develop diabetes based on diagnostic measurements. The goal is to build machine learning models to support early detection and improve healthcare outcomes.

🎯 Objectives

Explore the Pima Indians Diabetes dataset to identify patterns.

Perform data preprocessing and feature engineering.

Build classification models to predict diabetes occurrence.

Evaluate model performance and interpret results.

🗂️ Dataset

Source: UCI Machine Learning Repository

Features (8 numeric predictors + target):

Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skinfold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg / height in m²)

DiabetesPedigreeFunction: Diabetes pedigree function

Age: Age in years

Outcome: 0 = No diabetes, 1 = Diabetes (target)

Rows: 768

File format: .csv

🔧 Tools & Technologies

Python → Pandas, NumPy

Visualization → Matplotlib, Seaborn

Machine Learning → Scikit-learn (Logistic Regression, Random Forest, SVM, XGBoost)

Jupyter Notebook for analysis

📈 Methodology

Data Cleaning & Preprocessing

Handle missing / zero values in columns like Glucose, BloodPressure, BMI

Standardize / normalize numeric features

Split data into training and testing sets

Exploratory Data Analysis (EDA)

Distribution plots of features

Correlation heatmap

Compare feature values for diabetic vs non-diabetic patients

Feature Engineering

Create age groups / BMI categories if needed

Examine interactions (e.g., Glucose × BMI)

Model Building

Logistic Regression → baseline model

Random Forest / XGBoost → ensemble models for better accuracy

SVM → optional comparison

Evaluation Metrics

Accuracy, Precision, Recall, F1-score

ROC Curve & AUC

Confusion Matrix

📊 Key Insights

High Glucose and BMI levels are strong predictors of diabetes.

Older age and higher Diabetes Pedigree Function increase risk.

Ensemble models (Random Forest / XGBoost) performed better than simple logistic regression.

📌 Outcomes

Achieved accuracy ≈ 78–82% with Random Forest.

ROC-AUC ≈ 0.82 for best model.

Can be used for early screening and preventive recommendations.

🚀 Future Work

Deploy as a Streamlit app for interactive patient screening.

Use cross-validation and hyperparameter tuning for model improvement.

Integrate with larger datasets for better generalization.
