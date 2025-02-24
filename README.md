# Customer Churn Prediction - Project Report

## 1. Introduction

### 1.1 Project Overview

Customer churn prediction is a crucial task in business analytics to identify customers who are likely to leave a service. This project aims to build a machine learning model to predict customer churn based on historical data. The dataset contains various customer attributes such as account balance, credit score, tenure, and transaction history.

### 1.2 Objectives

To preprocess and clean the dataset for analysis.

To perform exploratory data analysis (EDA) to identify key patterns.

To build and evaluate machine learning models for predicting churn.

To optimize the model for better performance.

## 2. Data Preprocessing and Feature Engineering

### 2.1 Dataset Loading

The dataset is loaded using pandas (pd.read_csv). The dataset consists of numerical and categorical features.

### 2.2 Data Cleaning

Handling missing values (if any) using imputation techniques.

Removing duplicate records.

### 2.3 Feature Engineering

Categorical Encoding: One-hot encoding is applied to categorical features such as Geography.

New Features:

BalanceZero: A binary feature indicating if the customer has a zero balance.

BalanceToSalaryRatio: Derived as Balance / EstimatedSalary.

AgeGroup: Binning the age feature into categories (18-25, 26-35, etc.).

## 3. Exploratory Data Analysis (EDA)

Distribution of Churned vs. Non-Churned Customers

Correlation Analysis: Identifying relationships between numerical features.

Boxplots and Histograms: Understanding the spread of key features like Age, Balance, and Credit Score.

## 4. Model Selection and Training

### 4.1 Train-Test Split

The dataset is split into training and testing sets using train_test_split with an 80-20 ratio.

### 4.2 Machine Learning Models Used

RandomForestClassifier

GradientBoostingClassifier

### 4.3 Model Training

Each model is trained on the processed dataset using appropriate hyperparameters.

## 5. Model Evaluation

### 5.1 Performance Metrics Used

Accuracy Score

Precision, Recall, and F1-score

Confusion Matrix

ROC-AUC Score (if implemented)

### 5.2 Model Performance Summary

Random Forest:

Accuracy: ~86%

Precision, Recall, and F1-score for churn class were moderate.

Gradient Boosting:

Performed slightly better in recall, capturing more churned customers.

## 6. Conclusion

This project successfully implemented churn prediction using machine learning. Further improvements can be made by addressing class imbalance and tuning model parameters. Future work may involve deep learning models for better predictive accuracy.
